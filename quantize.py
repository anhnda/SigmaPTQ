"""
quantize.py  (group-wise clip-range patch)
==========================================
Weight-only group-wise RTN PTQ with GROUP-WISE clip-range selection.

CHANGE vs original: the clip range is now selected PER GROUP (matching the
group-wise quantizer), not per row. The quantizer and clip-range classes live
in base_cr_groupwise.py; this driver passes group_size into every prepare()
call and handles the per-group clip shape (d_out, n_groups, 1).

Clip-range methods (see base_cr_groupwise.py):
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

# Group-wise clip-range classes AND the matching per-group quantizer.
from base_cr import build_clip_range, make_quant_fn

try:
    from tqdm import tqdm
except ImportError:  # graceful no-op fallback if tqdm absent
    class _NoOpBar:
        def __init__(self, iterable=None):
            self._it = iterable
        def __iter__(self):
            return iter(self._it or [])
        def update(self, *a, **k):
            pass
        def set_postfix_str(self, *a, **k):
            pass
        def close(self):
            pass

    def tqdm(iterable=None, **kw):
        return _NoOpBar(iterable)


# ==========================================================================
# Slope function for sigma-aware metric (unchanged).
# ==========================================================================
def make_pre_act_fn(act: str = "silu"):
    act = act.lower()
    if act == "silu":
        def slope(A):
            s = torch.sigmoid(A)
            return s + A * s * (1 - s)
    elif act == "gelu":
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
# Layer metadata helpers.
# NOTE: up_proj output is NOT passed through SiLU in SwiGLU (only gate_proj is).
# Labeling up_proj as 'silu' applies a spurious slope weighting. Default here
# marks ONLY gate_proj as silu; flip USE_UP_PROJ_SILU to revert to the old
# (gate+up) behavior for an A/B.
# ==========================================================================
USE_UP_PROJ_SILU = False


def layer_activation(name: str) -> str:
    """Nonlinearity that follows this layer (for sigma slopes)."""
    n = name.lower()
    if "gate_proj" in n:
        return "silu"
    if "up_proj" in n and USE_UP_PROJ_SILU:
        return "silu"
    return "identity"


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


def run_forward(model, tokenizer, texts, n_samples, device, desc="calibration"):
    """One pass of calibration data through the model (hooks fire)."""
    n = min(n_samples, len(texts))
    with torch.no_grad():
        for i, text in enumerate(tqdm(texts[:n_samples], total=n, desc=desc,
                                      unit="sample", leave=False)):
            try:
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
                enc = {k: v.to(device) for k, v in enc.items()}
                _ = model(**enc, use_cache=False)
                if (i + 1) % 16 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                if i == 0:
                    print(f"  forward error: {str(e)[:120]}")
                continue


# ==========================================================================
# Per-group clip-fraction stat: clip is (d_out, n_groups, 1); compare each
# group's clip to that group's max-abs, average over rows and groups.
# ==========================================================================
def _clip_fraction(clip: torch.Tensor, Wf: torch.Tensor, group_size: int) -> float:
    d_out, d_in = Wf.shape
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * group_size, device=Wf.device, dtype=Wf.dtype)
        Wp[:, :d_in] = Wf
    else:
        Wp = Wf
    g_maxabs = Wp.reshape(d_out, n_groups, group_size).abs().amax(dim=2, keepdim=True)
    g_maxabs = g_maxabs.clamp(min=1e-8)                  # (d_out, G, 1)
    return (clip / g_maxabs).mean().item()


# ==========================================================================
# Stored-X hooks (unchanged).
# ==========================================================================
class XStoreHooks:
    def __init__(self, store, max_tokens):
        self.store = store
        self.max_tokens = max_tokens
        self.hooks = []

    def _hook(self, name):
        def hook(_m, inp, _o):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3 and x.shape[1] > self.max_tokens:
                idx = torch.randperm(x.shape[1], device=x.device)[:self.max_tokens]
                idx = idx.sort()[0]
                x = x[:, idx, :]
            self.store.setdefault(name, []).append(
                x.detach().reshape(-1, x.shape[-1]).cpu().float())
        return hook

    def register(self, model, names):
        for name, module in model.named_modules():
            if name in names:
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def _build_cr_for_layer(name, args):
    """Construct the clip-range object + extra prepare kwargs for a layer."""
    if args.clip_range == "sigma_aware":
        pre_act_fn = make_pre_act_fn(layer_activation(name))
        cr = build_clip_range("sigma_aware", lam=args.lam, n_grid=args.n_grid)
        return cr, dict(pre_act_fn=pre_act_fn)
    if args.clip_range == "linear_response":
        cr = build_clip_range("linear_response", n_grid=args.n_grid)
        return cr, dict()
    if args.clip_range == "mixed":
        if args.inner == "sigma":
            pre_act_fn = make_pre_act_fn(layer_activation(name))
            cr = build_clip_range("mixed", lam=args.lam, inner="sigma",
                                  n_grid=args.n_grid)
            return cr, dict(pre_act_fn=pre_act_fn)
        cr = build_clip_range("mixed", lam=args.lam, inner="linear",
                              n_grid=args.n_grid)
        return cr, dict()
    raise ValueError(f"{args.clip_range} should not use the stored-X backend")


@torch.no_grad()
def quantize_xstore_backend(model, tokenizer, calib_texts, linear_layers,
                            target_names, args):
    """Stored-X, layer-batched, GROUP-WISE clip for linear/mixed/sigma."""
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)
    gs = args.group_size
    bs = args.layer_batch_size
    groups = [linear_layers[i:i + bs] for i in range(0, len(linear_layers), bs)]

    stats = []
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for gi, group in enumerate(groups):
        batch_targets = {name for name, _ in group if name in target_names}
        store = {}
        if batch_targets:
            hooks = XStoreHooks(store, args.max_tokens_per_sample)
            hooks.register(model, batch_targets)
            run_forward(model, tokenizer, calib_texts, args.n_calib, device,
                        desc=f"X batch {gi+1}/{len(groups)}")
            hooks.remove()

        for name, module in group:
            if args.skip_lmhead and "lm_head" in name.lower():
                bar.update(1)
                bar.set_postfix_str(f"{name.split('.')[-1]} SKIPPED")
                continue
            W = module.weight.data
            orig_dtype = W.dtype
            Wf = W.float()

            X = None
            if name in store and store[name]:
                Xt = torch.cat(store[name], dim=0)            # (n_tok, d_in)
                if Xt.shape[0] > args.max_tok_total:
                    idx = torch.randperm(Xt.shape[0])[:args.max_tok_total]
                    Xt = Xt[idx]
                X = Xt.t().contiguous().to(device)            # (d_in, n_tok)

            if X is None:
                cr = build_clip_range("weight_mse", n_grid=args.n_grid)
                cr.prepare(Wf, group_size=gs)
            else:
                cr, extra = _build_cr_for_layer(name, args)
                cr.prepare(Wf, group_size=gs, X=X, **extra)

            clip = cr.select_clip(Wf, quant_fn)               # (d_out, G, 1)
            module.weight.data = quant_fn(Wf, clip).to(orig_dtype)

            frac = _clip_fraction(clip, Wf, gs)
            stats.append(frac)
            bar.update(1)
            bar.set_postfix_str(f"{name.split('.')[-1]} cf={frac:.3f}")

            if name in store:
                del store[name]
            del X
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        store.clear()
        gc.collect()

    bar.close()
    return stats


@torch.no_grad()
def quantize_weight_mse_backend(model, linear_layers, target_names, args):
    """weight_mse needs no activations: GROUP-WISE clip directly."""
    quant_fn = make_quant_fn(args.bits, args.group_size)
    gs = args.group_size
    stats = []
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for name, module in linear_layers:
        if args.skip_lmhead and "lm_head" in name.lower():
            bar.update(1)
            bar.set_postfix_str(f"{name.split('.')[-1]} SKIPPED")
            continue
        W = module.weight.data
        orig_dtype = W.dtype
        Wf = W.float()
        cr = build_clip_range(args.clip_range, n_grid=args.n_grid)
        cr.prepare(Wf, group_size=gs)
        clip = cr.select_clip(Wf, quant_fn)                   # (d_out, G, 1)
        module.weight.data = quant_fn(Wf, clip).to(orig_dtype)
        frac = _clip_fraction(clip, Wf, gs)
        stats.append(frac)
        bar.update(1)
        bar.set_postfix_str(f"{name.split('.')[-1]} cf={frac:.3f}")
    bar.close()
    return stats


# ==========================================================================
# Driver.
# ==========================================================================
@torch.no_grad()
def quantize_model(model, tokenizer, calib_texts, args):
    linear_layers = [(name, m) for name, m in model.named_modules()
                     if isinstance(m, nn.Linear)]
    target_names = {name for name, _ in linear_layers
                    if "lm_head" not in name.lower()}

    print(f"\nFound {len(linear_layers)} Linear layers. "
          f"Clip-range: {args.clip_range} | GROUP-WISE clip")

    if args.clip_range in ("rtn", "weight_mse"):
        print("Backend: weight-only (no activations).")
        stats = quantize_weight_mse_backend(model, linear_layers,
                                            target_names, args)
    else:
        print("Backend: stored-X, layer-batched.")
        stats = quantize_xstore_backend(model, tokenizer, calib_texts,
                                        linear_layers, target_names, args)

    if stats:
        print(f"\nClip selection done. Mean clip fraction across "
              f"layers/groups: {np.mean(stats):.3f}")


def main():
    p = argparse.ArgumentParser(
        description="Group-wise RTN weight-only PTQ with GROUP-WISE clip-range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-dir", type=str,
                   default="./quantized_models/model_rtn_cr")
    p.add_argument("--clip-range", type=str, default="linear_response",
                   choices=["rtn", "weight_mse", "linear_response", "mixed", "sigma_aware"])
    p.add_argument("--inner", type=str, default="linear", choices=["linear", "sigma"])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--bits", type=int, default=4, choices=[3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-grid", type=int, default=20)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-tokens-per-sample", type=int, default=512)
    p.add_argument("--layer-batch-size", type=int, default=16)
    p.add_argument("--max-tok-total", type=int, default=4096)
    p.add_argument("--calib-dataset", type=str, default="wikitext2",
                   choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", type=str, default="./calibration_cache")
    p.add_argument("--skip-lmhead", dest="skip_lmhead", action="store_true",
                   default=True)
    p.add_argument("--quant-lmhead", dest="skip_lmhead", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--up-proj-silu", dest="up_proj_silu", action="store_true",
                   default=False,
                   help="Treat up_proj as silu-followed (old behavior). Default "
                        "False: only gate_proj gets slope weighting.")

    args = p.parse_args()

    global USE_UP_PROJ_SILU
    USE_UP_PROJ_SILU = args.up_proj_silu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("Group-wise RTN PTQ  |  clip-range =", args.clip_range,
          "| bits =", args.bits, "| group =", args.group_size,
          "| GROUP-WISE clip")
    print("Model:", args.model_path, "| device:", args.device)
    print("up_proj treated as silu:", USE_UP_PROJ_SILU)
    print("Skip lm_head quantization:", args.skip_lmhead)
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