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
# Layer metadata helpers
# ==========================================================================
def layer_activation(name: str) -> str:
    """Nonlinearity that follows this layer (for sigma slopes)."""
    n = name.lower()
    if "gate_proj" in n or "up_proj" in n:
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
# BACKEND A : Gram accumulation (Tier 1/2/3).
#   One forward pass. Hooks accumulate G = XX^T/N per layer incrementally;
#   activations are never stored. Token-count-independent (d_in^2 per layer).
#   If --gram-layer-batch > 0, layers are processed in groups to cap the
#   number of live Gram matrices (trades passes for RAM).
# ==========================================================================
class GramHooks:
    """Registers hooks that fold each layer's activations into a GramAccumulator."""

    def __init__(self, accums, max_tokens, gram_device):
        # accums: dict name -> GramAccumulator
        self.accums = accums
        self.max_tokens = max_tokens
        self.gram_device = gram_device
        self.hooks = []

    def _hook(self, name):
        acc = self.accums[name]

        def hook(_m, inp, _o):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3 and x.shape[1] > self.max_tokens:
                idx = torch.randperm(x.shape[1], device=x.device)[:self.max_tokens]
                idx = idx.sort()[0]
                x = x[:, idx, :]
            Xc = x.detach().reshape(-1, x.shape[-1]).t().contiguous()  # (d_in, n)
            acc.update(Xc)        # accumulator handles device/dtype move
        return hook

    def register(self, model, names):
        for name, module in model.named_modules():
            if name in names:
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


@torch.no_grad()
def quantize_gram_backend(model, tokenizer, calib_texts, linear_layers,
                          target_names, args):
    """Tier 1/2/3 path: accumulate Gram, then per-layer clip search + RTN."""
    from base_cr import GramAccumulator
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)
    needs_acts = args.clip_range in ("linear_response", "mixed")  # weight_mse: none
    # Gram matrices live on CPU by default to save VRAM; scoring moves the one
    # active layer's Gram to GPU.
    gram_device = "cpu" if args.gram_on_cpu else device

    # Determine which layers are processed together. If gram_layer_batch<=0,
    # all layers' Grams are accumulated in a single pass (max RAM, 1 pass).
    if not needs_acts:
        groups = [linear_layers]  # weight_mse: no activations, single trivial group
    elif args.gram_layer_batch and args.gram_layer_batch > 0:
        groups = [linear_layers[i:i + args.gram_layer_batch]
                  for i in range(0, len(linear_layers), args.gram_layer_batch)]
    else:
        groups = [linear_layers]

    stats = []
    quantized = 0
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for gi, group in enumerate(groups):
        # Build accumulators only for layers in this group that need acts and
        # actually collect activations (skip lm_head etc.).
        accums = {}
        if needs_acts:
            for name, module in group:
                if name in target_names:
                    d_in = module.weight.shape[1]
                    accums[name] = GramAccumulator(d_in, gram_device, torch.float32)
            if accums:
                hooks = GramHooks(accums, args.max_tokens_per_sample, gram_device)
                hooks.register(model, set(accums.keys()))
                if len(groups) > 1:
                    print(f"  [Gram pass {gi+1}/{len(groups)}] "
                          f"accumulating {len(accums)} layers...")
                else:
                    print(f"  [Gram pass] accumulating {len(accums)} layers "
                          "in a single pass...")
                run_forward(model, tokenizer, calib_texts, args.n_calib, device,
                            desc=f"Gram pass {gi+1}/{len(groups)}"
                            if len(groups) > 1 else "Gram pass")
                hooks.remove()

        # Quantize every layer in this group.
        for name, module in group:
            W = module.weight.data
            orig_dtype = W.dtype
            Wf = W.float()

            G = None
            if needs_acts and name in accums:
                G = accums[name].finalize().to(device)

            if needs_acts and G is None:
                # layer had no activations (e.g. lm_head) -> weight_mse fallback
                cr = build_clip_range("weight_mse", n_grid=args.n_grid)
                cr.prepare(Wf)
            else:
                cr = build_clip_range(args.clip_range, lam=args.lam,
                                      inner=args.inner, n_grid=args.n_grid)
                cr.prepare(Wf, G=G)

            clip = cr.select_clip(Wf, quant_fn)
            module.weight.data = quant_fn(Wf, clip).to(orig_dtype)

            frac = (clip.squeeze(1) /
                    Wf.abs().amax(dim=1).clamp(min=1e-8)).mean().item()
            stats.append(frac)
            quantized += 1
            bar.update(1)
            bar.set_postfix_str(f"{name.split('.')[-1]} cf={frac:.3f}")

            del G
            if name in accums:
                del accums[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accums.clear()
        gc.collect()

    bar.close()
    return stats


# ==========================================================================
# BACKEND B : stored-X, layer-batched (sigma-aware).
#   Per-row slopes prevent a single shared Gram, so activations are stored.
#   Layers are processed in batches: collect X for the batch in one pass,
#   quantize, free, repeat. RAM is bounded by batch_size * per-layer-X.
# ==========================================================================
class XStoreHooks:
    def __init__(self, store, max_tokens):
        self.store = store        # name -> list of (n, d_in) cpu float
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
    """Construct the clip-range object for a layer needing stored X."""
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
def quantize_sigma_backend(model, tokenizer, calib_texts, linear_layers,
                           target_names, args):
    """Sigma-aware path: stored-X with layer batching."""
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)
    bs = args.layer_batch_size
    groups = [linear_layers[i:i + bs] for i in range(0, len(linear_layers), bs)]

    stats = []
    quantized = 0
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for gi, group in enumerate(groups):
        batch_targets = {name for name, _ in group if name in target_names}
        store = {}
        if batch_targets:
            hooks = XStoreHooks(store, args.max_tokens_per_sample)
            hooks.register(model, batch_targets)
            run_forward(model, tokenizer, calib_texts, args.n_calib, device,
                        desc=f"sigma batch {gi+1}/{len(groups)}")
            hooks.remove()

        for name, module in group:
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
                cr.prepare(Wf)
            else:
                pre_act_fn = make_pre_act_fn(layer_activation(name))
                cr = build_clip_range("sigma_aware", lam=args.lam,
                                      n_grid=args.n_grid)
                cr.prepare(Wf, X=X, pre_act_fn=pre_act_fn)

            clip = cr.select_clip(Wf, quant_fn)
            module.weight.data = quant_fn(Wf, clip).to(orig_dtype)

            frac = (clip.squeeze(1) /
                    Wf.abs().amax(dim=1).clamp(min=1e-8)).mean().item()
            stats.append(frac)
            quantized += 1
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
def quantize_xstore_backend(model, tokenizer, calib_texts, linear_layers,
                            target_names, args):
    """Stored-X with layer batching for linear_response / mixed / sigma_aware."""
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)
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
                cr.prepare(Wf)
            else:
                cr, extra = _build_cr_for_layer(name, args)
                cr.prepare(Wf, X=X, **extra)

            clip = cr.select_clip(Wf, quant_fn)
            module.weight.data = quant_fn(Wf, clip).to(orig_dtype)

            frac = (clip.squeeze(1) /
                    Wf.abs().amax(dim=1).clamp(min=1e-8)).mean().item()
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
# ==========================================================================
# Driver: route to the correct backend.
# ==========================================================================
@torch.no_grad()
def quantize_model(model, tokenizer, calib_texts, args):
    linear_layers = [(name, m) for name, m in model.named_modules()
                     if isinstance(m, nn.Linear)]
    target_names = {name for name, _ in linear_layers
                    if "lm_head" not in name.lower()}

    print(f"\nFound {len(linear_layers)} Linear layers. "
          f"Clip-range: {args.clip_range}")

    if args.clip_range == "weight_mse":
        print("Backend: weight-only (no activations).")
        stats = quantize_weight_mse_backend(model, linear_layers,
                                            target_names, args)
    else:
        print("Backend: stored-X, layer-batched.")
        stats = quantize_xstore_backend(model, tokenizer, calib_texts,
                                        linear_layers, target_names, args)

    if stats:
        print(f"\nClip selection done. Mean clip fraction across layers: "
              f"{np.mean(stats):.3f}")


@torch.no_grad()
def quantize_weight_mse_backend(model, linear_layers, target_names, args):
    """weight_mse needs no activations: quantize directly."""
    quant_fn = make_quant_fn(args.bits, args.group_size)
    stats = []
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for name, module in linear_layers:
        W = module.weight.data
        orig_dtype = W.dtype
        Wf = W.float()
        cr = build_clip_range("weight_mse", n_grid=args.n_grid)
        cr.prepare(Wf)
        clip = cr.select_clip(Wf, quant_fn)
        module.weight.data = quant_fn(Wf, clip).to(orig_dtype)
        frac = (clip.squeeze(1) /
                Wf.abs().amax(dim=1).clamp(min=1e-8)).mean().item()
        stats.append(frac)
        bar.update(1)
        bar.set_postfix_str(f"{name.split('.')[-1]} cf={frac:.3f}")
    bar.close()
    return stats
@torch.no_grad()
def quantize_mixed_sigma_backend(model, tokenizer, calib_texts, linear_layers,
                                 target_names, args):
    """mixed + inner=sigma: stored-X batched, shared mean-slope metric."""
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)
    bs = args.layer_batch_size
    groups = [linear_layers[i:i + bs] for i in range(0, len(linear_layers), bs)]
    stats = []
    quantized = 0
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for gi, group in enumerate(groups):
        batch_targets = {name for name, _ in group if name in target_names}
        store = {}
        if batch_targets:
            hooks = XStoreHooks(store, args.max_tokens_per_sample)
            hooks.register(model, batch_targets)
            run_forward(model, tokenizer, calib_texts, args.n_calib, device,
                        desc=f"mixed-sigma batch {gi+1}/{len(groups)}")
            hooks.remove()
        for name, module in group:
            W = module.weight.data
            orig_dtype = W.dtype
            Wf = W.float()
            X = None
            if name in store and store[name]:
                Xt = torch.cat(store[name], dim=0)
                if Xt.shape[0] > args.max_tok_total:
                    idx = torch.randperm(Xt.shape[0])[:args.max_tok_total]
                    Xt = Xt[idx]
                X = Xt.t().contiguous().to(device)
            if X is None:
                cr = build_clip_range("weight_mse", n_grid=args.n_grid)
                cr.prepare(Wf)
            else:
                pre_act_fn = make_pre_act_fn(layer_activation(name))
                cr = build_clip_range("mixed", lam=args.lam, inner="sigma",
                                      n_grid=args.n_grid)
                cr.prepare(Wf, X=X, pre_act_fn=pre_act_fn)
            clip = cr.select_clip(Wf, quant_fn)
            module.weight.data = quant_fn(Wf, clip).to(orig_dtype)
            frac = (clip.squeeze(1) /
                    Wf.abs().amax(dim=1).clamp(min=1e-8)).mean().item()
            stats.append(frac); quantized += 1
            bar.update(1)
            bar.set_postfix_str(f"{name.split('.')[-1]} cf={frac:.3f}")
            if name in store:
                del store[name]
            del X
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        store.clear(); gc.collect()
    bar.close()
    return stats

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
    p.add_argument("--max-tokens-per-sample", type=int, default=512,
                   help="Tokens kept per sample at the hook (subsample).")
    # --- memory control ---
    p.add_argument("--gram-layer-batch", type=int, default=0,
                   help="Gram backend (Tier1/2/3): layers per accumulation pass. "
                        "0 = all layers in ONE pass (max RAM ~ sum of d_in^2, "
                        "but token-count-independent). >0 trades extra passes "
                        "for lower peak RAM (caps live Gram matrices).")
    p.add_argument("--gram-on-cpu", action="store_true", default=True,
                   help="Hold Gram matrices on CPU (default); scoring moves the "
                        "active layer's Gram to GPU. Saves VRAM.")
    p.add_argument("--gram-on-gpu", dest="gram_on_cpu", action="store_false",
                   help="Keep Gram matrices on GPU (faster, more VRAM).")
    p.add_argument("--layer-batch-size", type=int, default=16,
                   help="Sigma backend: layers per X-collection batch. "
                        "Caps CPU RAM at batch_size * per-layer stored X.")
    p.add_argument("--max-tok-total", type=int, default=4096,
                   help="Sigma backend: max total tokens kept per layer in X "
                        "(stored-X memory scales linearly with this).")
    p.add_argument("--calib-dataset", type=str, default="wikitext2",
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