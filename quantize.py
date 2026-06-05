"""
quantize.py
===========
Naive clip-range PTQ with group-wise RTN (Round-To-Nearest), weight-only.

Pipeline (mirrors the AWQ example's structure):
  1. Load model + tokenizer.
  2. Collect per-layer calibration activations X (one hook pass).
  3. For each Linear layer, pick a per-row clip range with the selected
     clip-range method, then group-wise (default 128) symmetric RTN quantize.
  4. Save the (fake-)quantized model.

Quantization grid (paper Eq. 4, symmetric):
      Delta = c / (2^{b-1} - 1)
      Q_c(w) = Delta * clip(round(w/Delta), -(2^{b-1}-1), 2^{b-1}-1)
  done PER GROUP of `group_size` input channels, with the per-row clip `c`
  chosen by the clip-range method, applied within each group's max-abs scale.

Clip-range methods (see base_cr.py):
  weight_mse        Tier 1 : M = I
  linear_response   Tier 2 : M = X X^T / N
  mixed             Tier 3 : M = (1-lam) I + lam * inner   (inner=linear|sigma)
  sigma_aware       Paper  : M = X S^2 X^T / N  (+ optional identity floor)

Usage:
  python quantize.py --model-path ./models/Mistral-7B-v0.3 \
                     --clip-range sigma_aware --lam 0.9 \
                     --bits 4 --group-size 128 --output-dir ./quantized_models/out
"""

from __future__ import annotations

import os
import gc
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from base_cr import build_clip_range


# ==========================================================================
# Group-wise symmetric RTN quantizer
# ==========================================================================
@torch.no_grad()
def groupwise_rtn_symmetric(W: torch.Tensor, clip: torch.Tensor,
                            bits: int, group_size: int) -> torch.Tensor:
    """
    Group-wise symmetric RTN with a per-row clip ceiling.

    W    : (d_out, d_in)
    clip : (d_out, 1) per-row clip magnitude c (from clip-range search).
           Each group's scale is set from min(group_max_abs, c) so the clip
           acts as a saturation ceiling on the per-row tail.
    bits : bit width.
    group_size : input channels per group.

    returns : dequantized W of shape (d_out, d_in).
    """
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1          # symmetric range [-qmax, qmax]

    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * group_size, device=W.device, dtype=W.dtype)
        Wp[:, :d_in] = W
    else:
        Wp = W

    Wg = Wp.reshape(d_out, n_groups, group_size)             # (d_out, G, gs)

    # Per-group max-abs, capped by the per-row clip c.
    g_maxabs = Wg.abs().amax(dim=2, keepdim=True)            # (d_out, G, 1)
    c = clip.unsqueeze(1)                                    # (d_out, 1, 1)
    eff = torch.minimum(g_maxabs, c).clamp(min=1e-8)         # saturation ceiling

    delta = eff / qmax                                       # (d_out, G, 1)
    q = torch.round(Wg / delta).clamp(-qmax, qmax)
    Wdq = q * delta

    Wdq = Wdq.reshape(d_out, n_groups * group_size)
    if pad > 0:
        Wdq = Wdq[:, :d_in]
    return Wdq


def make_quant_fn(bits: int, group_size: int):
    """Return a QuantFn closure with bits/group_size baked in."""
    def quant_fn(W: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        return groupwise_rtn_symmetric(W, clip, bits, group_size)
    return quant_fn


# ==========================================================================
# Slope function for sigma-aware metric.
# Choose the nonlinearity that follows the layer (GELU / SiLU). Default SiLU
# (Llama/Mistral SwiGLU gate/up). For non-MLP layers sigma' = 1 (identity)
# recovers the linear-response metric (paper Cor. 1).
# ==========================================================================
def make_pre_act_fn(act: str = "silu"):
    act = act.lower()
    if act == "silu":
        # SiLU(a) = a*sigmoid(a);  SiLU'(a) = sigmoid(a) + a*sigmoid(a)*(1-sigmoid(a))
        def slope(A):
            s = torch.sigmoid(A)
            return s + A * s * (1 - s)
    elif act == "gelu":
        # tanh-approx GELU derivative
        def slope(A):
            c = 0.7978845608028654  # sqrt(2/pi)
            k = 0.044715
            inner = c * (A + k * A.pow(3))
            t = torch.tanh(inner)
            dt = (1 - t.pow(2)) * c * (1 + 3 * k * A.pow(2))
            return 0.5 * (1 + t) + 0.5 * A * dt
    elif act == "identity":
        def slope(A):
            return torch.ones_like(A)
    else:
        raise ValueError(f"Unknown activation {act!r}")
    return slope


# ==========================================================================
# Activation collection
# ==========================================================================
class ActCollector:
    def __init__(self, model, max_tokens_per_sample=512, device="cuda"):
        self.model = model
        self.acts = {}          # name -> list of (n_tok_sub, d_in) cpu float
        self.hooks = []
        self.max_tokens = max_tokens_per_sample
        self.device = device

    def _hook(self, name):
        def hook(_m, inp, _o):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3 and x.shape[1] > self.max_tokens:
                idx = torch.randperm(x.shape[1], device=x.device)[:self.max_tokens]
                idx = idx.sort()[0]
                x = x[:, idx, :]
            self.acts.setdefault(name, []).append(
                x.detach().reshape(-1, x.shape[-1]).cpu().float()
            )
        return hook

    def collect(self, tokenizer, texts, n_samples, target_names):
        for name, module in self.model.named_modules():
            if name in target_names:
                self.hooks.append(module.register_forward_hook(self._hook(name)))

        with torch.no_grad():
            for i, text in enumerate(texts[:n_samples]):
                try:
                    enc = tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=512)
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    _ = self.model(**enc, use_cache=False)
                    if (i + 1) % 16 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    if i == 0:
                        print(f"  forward error: {str(e)[:120]}")
                    continue

        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get(self, name, device, max_tok=4096):
        """Return X of shape (d_in, n_tok) on `device`, subsampled to max_tok."""
        if name not in self.acts or not self.acts[name]:
            return None
        Xt = torch.cat(self.acts[name], dim=0)        # (n_tok, d_in)
        if Xt.shape[0] > max_tok:
            idx = torch.randperm(Xt.shape[0])[:max_tok]
            Xt = Xt[idx]
        return Xt.t().contiguous().to(device)         # (d_in, n_tok)

    def clear(self):
        self.acts = {}
        gc.collect()


# ==========================================================================
# Layer-name -> activation nonlinearity mapping.
# For Llama/Mistral SwiGLU: gate_proj & up_proj feed SiLU. Others -> identity.
# ==========================================================================
def layer_activation(name: str) -> str:
    n = name.lower()
    if "gate_proj" in n or "up_proj" in n:
        return "silu"
    return "identity"


# ==========================================================================
# Calibration loader (lightweight WikiText-2; reuse paper's utils if present)
# ==========================================================================
def load_calib(tokenizer, dataset, n_samples, seqlen, seed, cache_dir):
    try:
        from calibration_utils import (get_c4_calibration_data,
                                        get_wikitext2_calibration_data)
        if dataset == "c4":
            return get_c4_calibration_data(tokenizer, n_samples, seqlen, seed,
                                           cache_dir=cache_dir)
        if dataset in ("wikitext2", "wikitext"):
            return get_wikitext2_calibration_data(tokenizer, n_samples, seqlen,
                                                  seed, cache_dir=cache_dir)
    except Exception as e:
        print(f"  calibration_utils unavailable ({str(e)[:80]}); "
              "falling back to simple wikitext loader.")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 0]
    return texts[:n_samples]


# ==========================================================================
# Main quantization driver
# ==========================================================================
@torch.no_grad()
def quantize_model(model, tokenizer, calib_texts, args):
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)

    linear_layers = [(name, m) for name, m in model.named_modules()
                     if isinstance(m, nn.Linear)]
    # Skip lm_head by default for activation collection unless requested;
    # it is still quantized with weight_mse (no activations needed) if absent.
    target_names = {name for name, _ in linear_layers
                    if "lm_head" not in name.lower()}

    print(f"\nFound {len(linear_layers)} Linear layers. "
          f"Clip-range method: {args.clip_range}")

    needs_acts = args.clip_range in ("linear_response", "mixed", "sigma_aware")

    collector = ActCollector(model, args.max_tokens_per_sample, device)
    if needs_acts:
        print("Collecting calibration activations (single hook pass)...")
        collector.collect(tokenizer, calib_texts, args.n_calib, target_names)

    stats = []
    for idx, (name, module) in enumerate(linear_layers):
        W = module.weight.data
        orig_dtype = W.dtype
        Wf = W.float()

        # Pick the clip-range method. lm_head / layers w/o activations fall
        # back to weight_mse so the pipeline never stalls.
        X = collector.get(name, device) if needs_acts else None
        if needs_acts and X is None:
            cr = build_clip_range("weight_mse", n_grid=args.n_grid)
            cr.prepare(None, Wf)
        else:
            inner = args.inner
            pre_act_fn = None
            if args.clip_range == "sigma_aware":
                pre_act_fn = make_pre_act_fn(layer_activation(name))
            elif args.clip_range == "mixed" and inner == "sigma":
                pre_act_fn = make_pre_act_fn(layer_activation(name))
            cr = build_clip_range(args.clip_range, lam=args.lam, inner=inner,
                                  n_grid=args.n_grid)
            cr.prepare(X.to(torch.float32) if X is not None else None, Wf,
                       pre_act_fn=pre_act_fn)

        clip = cr.select_clip(Wf, quant_fn)           # (d_out, 1)
        Wq = quant_fn(Wf, clip).to(orig_dtype)
        module.weight.data = Wq

        mean_frac = (clip.squeeze(1) /
                     Wf.abs().amax(dim=1).clamp(min=1e-8)).mean().item()
        stats.append(mean_frac)

        if idx < 3 or (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(linear_layers)}] {name}: "
                  f"mean clip frac={mean_frac:.3f}")

        del X
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collector.clear()
    if stats:
        print(f"\nClip selection done. Mean clip fraction across layers: "
              f"{np.mean(stats):.3f}")


def main():
    p = argparse.ArgumentParser(
        description="Group-wise RTN weight-only PTQ with clip-range selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-dir", type=str,
                   default="./quantized_models/model_rtn_cr")
    p.add_argument("--clip-range", type=str, default="linear_response",
                   choices=["weight_mse", "linear_response", "mixed", "sigma_aware"],
                   help="Clip-range metric tier.")
    p.add_argument("--inner", type=str, default="linear", choices=["linear", "sigma"],
                   help="Inner metric for 'mixed' tier.")
    p.add_argument("--lam", type=float, default=0.5,
                   help="Floor weight lambda for mixed / floored sigma_aware "
                        "(M = (1-lam) I + lam M_inner). lam=0 -> unfloored.")
    p.add_argument("--bits", type=int, default=4, choices=[3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-grid", type=int, default=20)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-tokens-per-sample", type=int, default=512)
    p.add_argument("--calib-dataset", type=str, default="c4",
                   choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", type=str, default="./calibration_cache")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("Group-wise RTN PTQ  |  clip-range =", args.clip_range,
          "| bits =", args.bits, "| group =", args.group_size)
    print("Model:", args.model_path, "| device:", args.device)
    print("=" * 78)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    calib_texts = load_calib(tokenizer, args.calib_dataset, args.n_calib,
                             args.seqlen, args.seed, args.cache_dir)

    quantize_model(model, tokenizer, calib_texts, args)

    print(f"\nSaving quantized model to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DONE.")


if __name__ == "__main__":
    main()