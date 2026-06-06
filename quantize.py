"""
quantize.py
===========
Weight-only group-wise RTN PTQ with local-Jacobian (NAC) clip-range metrics.

The metric is now ROLE-AWARE and the SwiGLU gate/up projections are quantized
as a COUPLED PAIR (paper Sec. 4.2), which the old per-layer prepare() could
not express. Routing:

    --metric weight_mse        M=I, no activations
    --metric linear_response   M_2 for every layer
    --metric pointwise         M_sigma for gate (incomplete), M_2 elsewhere
    --metric gated             gate->Eq.11, up->Eq.13 (+Eq.14 rho), rest->M_2
    --metric mixed             (1-lam) I + lam * (role metric)

For "gated", gate and up are processed inside the same block using one X
capture and the shared g=W_g X, u=W_u X. down_proj and attention use M_2.

Usage:
  python quantize.py --model-path ./models/Mistral-7B-v0.3 \
      --metric gated --bits 3 --group-size 128 --use-rho \
      --output-dir ./quantized_models/mistral_gated_w3
"""

from __future__ import annotations

import os
import gc
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

import metric  # noqa: F401  (kept for parity / debugging)
import sensitivity as sens
from quantizer import make_quant_fn
from clip_search import select_clip, rtn_clip
from coupling import build_score_fn, SwiGLUBlock
import layer_registry as reg

try:
    from tqdm import tqdm
except ImportError:
    class _NoOpBar:
        def __init__(self, iterable=None, **k): self._it = iterable
        def __iter__(self): return iter(self._it or [])
        def update(self, *a, **k): pass
        def set_postfix_str(self, *a, **k): pass
        def close(self): pass
    def tqdm(iterable=None, **kw): return _NoOpBar(iterable)


# --------------------------------------------------------------------------
# Stored-X hooks (carried over from the original).
# --------------------------------------------------------------------------
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
        print(f"  calibration_utils unavailable ({str(e)[:80]}); wikitext fallback.")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 0]
    return texts[:n_samples]


def run_forward(model, tokenizer, texts, n_samples, device, desc="calibration"):
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


def _clip_fraction(clip, Wf, group_size):
    d_out, d_in = Wf.shape
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * group_size, device=Wf.device, dtype=Wf.dtype)
        Wp[:, :d_in] = Wf
    else:
        Wp = Wf
    g_maxabs = Wp.reshape(d_out, n_groups, group_size).abs().amax(dim=2, keepdim=True)
    g_maxabs = g_maxabs.clamp(min=1e-8)
    return (clip / g_maxabs).mean().item()


def _xfrom_store(store, name, device, max_tok_total):
    if name not in store or not store[name]:
        return None
    Xt = torch.cat(store[name], dim=0)             # (n_tok, d_in)
    if Xt.shape[0] > max_tok_total:
        idx = torch.randperm(Xt.shape[0])[:max_tok_total]
        Xt = Xt[idx]
    return Xt.t().contiguous().to(device)          # (d_in, n_tok)


# --------------------------------------------------------------------------
# Per-layer score routing for the non-coupled roles (linear/pointwise/mixed).
# --------------------------------------------------------------------------
def _solo_score_fn(name, Wf, X, args):
    """Score for a layer NOT handled by SwiGLU coupling."""
    gs = args.group_size
    role = reg.role_of(name)

    if X is None:  # no activations captured -> weight_mse
        return build_score_fn(Wf, group_size=gs, sensitivity=None)

    if args.metric == "weight_mse":
        return build_score_fn(Wf, group_size=gs, sensitivity=None)

    if args.metric == "linear_response":
        return build_score_fn(Wf, group_size=gs, X=X,
                              sensitivity=sens.ConstantSensitivity())

    if args.metric == "pointwise":
        # M_sigma only where there is a pointwise nonlinearity (gate). Other
        # roles fall back to M_2.
        if role == "gate":
            A = Wf @ X
            s = sens.PointwiseSensitivity(A, act="silu")
            return build_score_fn(Wf, group_size=gs, X=X, sensitivity=s,
                                  lam=args.lam)
        return build_score_fn(Wf, group_size=gs, X=X,
                              sensitivity=sens.ConstantSensitivity())

    if args.metric in ("gated", "mixed"):
        # gate/up handled by coupler elsewhere; here only down/attn/other.
        # Use M_2 (identity geometry) for these per paper G3 / down.
        return build_score_fn(Wf, group_size=gs, X=X,
                              sensitivity=sens.ConstantSensitivity())

    raise ValueError(f"Unknown metric {args.metric!r}")


@torch.no_grad()
def quantize_xstore_backend(model, tokenizer, calib_texts, linear_layers,
                            target_names, args):
    device = args.device
    quant_fn = make_quant_fn(args.bits, args.group_size)
    gs = args.group_size
    bs = args.layer_batch_size

    mlp_blocks = reg.group_mlp_blocks(linear_layers)
    # Map a layer name -> its block key (for gate/up coupled handling).
    name_to_block = {}
    for key, parts in mlp_blocks.items():
        for role in ("gate", "up", "down"):
            if role in parts:
                name_to_block[parts[role][0]] = key

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

        # Track which coupled blocks are fully ready within this batch.
        done_coupled = set()

        for name, module in group:
            if args.skip_lmhead and "lm_head" in name.lower():
                bar.update(1); bar.set_postfix_str(f"{name.split('.')[-1]} SKIPPED")
                continue

            role = reg.role_of(name)
            Wf = module.weight.data.float()
            orig_dtype = module.weight.data.dtype
            X = _xfrom_store(store, name, device, args.max_tok_total)

            # ---- Coupled SwiGLU gate/up under the gated/mixed metric ----
            coupled = (args.metric in ("gated", "mixed") and role in ("gate", "up"))
            key = name_to_block.get(name)
            if coupled and key in mlp_blocks and key not in done_coupled:
                parts = mlp_blocks[key]
                if all(r in parts for r in ("gate", "up", "down")):
                    g_name, g_mod = parts["gate"]
                    u_name, u_mod = parts["up"]
                    d_name, d_mod = parts["down"]
                    Xg = _xfrom_store(store, g_name, device, args.max_tok_total)
                    Xu = _xfrom_store(store, u_name, device, args.max_tok_total)
                    Xshared = Xg if Xg is not None else Xu
                    if Xshared is not None:
                        Wg = g_mod.weight.data.float()
                        Wu = u_mod.weight.data.float()
                        Wd = d_mod.weight.data.float()
                        blk = SwiGLUBlock(Wg, Wu, Wd, Xshared,
                                          act="silu", use_rho=args.use_rho,
                                          power=args.power,
                                          random_seed=(args.seed
                                                       if args.random_sens
                                                       else None))
                        lam = args.lam if args.metric == "mixed" else 0.0
                        # gate
                        s_g = blk.gate_score_fn(gs, lam)
                        clip_g = select_clip(Wg, quant_fn, s_g, group_size=gs,
                                             n_grid=args.n_grid,
                                             grid_min=args.grid_min, grid_max=args.grid_max)
                        g_mod.weight.data = quant_fn(Wg, clip_g).to(g_mod.weight.data.dtype)
                        stats.append(_clip_fraction(clip_g, Wg, gs))
                        # up
                        s_u = blk.up_score_fn(gs, lam)
                        clip_u = select_clip(Wu, quant_fn, s_u, group_size=gs,
                                             n_grid=args.n_grid,
                                             grid_min=args.grid_min, grid_max=args.grid_max)
                        u_mod.weight.data = quant_fn(Wu, clip_u).to(u_mod.weight.data.dtype)
                        stats.append(_clip_fraction(clip_u, Wu, gs))
                        done_coupled.add(key)
                        # advance bar for whichever of gate/up is in THIS group;
                        # the sibling may be in another group and will be skipped
                        # via done_coupled when reached.
                        bar.update(1)
                        bar.set_postfix_str(f"{name.split('.')[-1]} coupled "
                                            f"cf_g={stats[-2]:.3f} cf_u={stats[-1]:.3f}")
                        del Xg, Xu, Xshared
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

            # If this layer is the sibling of an already-coupled block, skip.
            if coupled and key in done_coupled:
                bar.update(1); bar.set_postfix_str(f"{name.split('.')[-1]} (coupled)")
                continue

            # ---- Solo path (linear/pointwise/weight_mse, or down/attn) ----
            score_fn = _solo_score_fn(name, Wf, X, args)
            if X is None and args.metric != "weight_mse":
                clip = rtn_clip(Wf, gs)  # nothing captured; fall back to RTN
            else:
                clip = select_clip(Wf, quant_fn, score_fn, group_size=gs,
                                   n_grid=args.n_grid,
                                   grid_min=args.grid_min, grid_max=args.grid_max)
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

        store.clear(); gc.collect()

    bar.close()
    return stats


@torch.no_grad()
def quantize_weight_mse_backend(model, linear_layers, args):
    quant_fn = make_quant_fn(args.bits, args.group_size)
    gs = args.group_size
    stats = []
    bar = tqdm(total=len(linear_layers), desc="Quantizing layers", unit="layer")
    for name, module in linear_layers:
        if args.skip_lmhead and "lm_head" in name.lower():
            bar.update(1); bar.set_postfix_str(f"{name.split('.')[-1]} SKIPPED")
            continue
        Wf = module.weight.data.float()
        orig_dtype = module.weight.data.dtype
        if args.metric == "rtn":
            clip = rtn_clip(Wf, gs)
        else:
            score_fn = build_score_fn(Wf, group_size=gs, sensitivity=None)
            clip = select_clip(Wf, quant_fn, score_fn, group_size=gs,
                               n_grid=args.n_grid,
                               grid_min=args.grid_min, grid_max=args.grid_max)
        module.weight.data = quant_fn(Wf, clip).to(orig_dtype)
        frac = _clip_fraction(clip, Wf, gs)
        stats.append(frac)
        bar.update(1); bar.set_postfix_str(f"{name.split('.')[-1]} cf={frac:.3f}")
    bar.close()
    return stats


@torch.no_grad()
def quantize_model(model, tokenizer, calib_texts, args):
    linear_layers = [(name, m) for name, m in model.named_modules()
                     if isinstance(m, nn.Linear)]
    target_names = {name for name, _ in linear_layers
                    if "lm_head" not in name.lower()}
    print(f"\nFound {len(linear_layers)} Linear layers. Metric: {args.metric} "
          f"| GROUP-WISE clip | rho={args.use_rho}")

    if args.metric in ("rtn", "weight_mse"):
        print("Backend: weight-only (no activations).")
        stats = quantize_weight_mse_backend(model, linear_layers, args)
    else:
        print("Backend: stored-X, layer-batched, role-aware (coupled SwiGLU).")
        stats = quantize_xstore_backend(model, tokenizer, calib_texts,
                                        linear_layers, target_names, args)
    if stats:
        print(f"\nClip selection done. Mean clip fraction: {np.mean(stats):.3f}")


def main():
    p = argparse.ArgumentParser(
        description="Group-wise RTN PTQ with local-Jacobian (NAC) metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-dir", type=str, default="./quantized_models/out")
    p.add_argument("--metric", type=str, default="gated",
                   choices=["rtn", "weight_mse", "linear_response",
                            "pointwise", "gated", "mixed"])
    p.add_argument("--lam", type=float, default=0.0,
                   help="identity floor for mixed/pointwise: (1-lam)I + lam*M.")
    p.add_argument("--use-rho", dest="use_rho", action="store_true", default=True,
                   help="apply downstream gain rho_k=||W_d^{(:,k)}||^2 (Eq.14).")
    p.add_argument("--no-rho", dest="use_rho", action="store_false")
    p.add_argument("--power", type=int, default=2, choices=[1, 2],
                   help="C1 ablation: 2 -> XS^2X^T (correct); 1 -> XSX^T control.")
    p.add_argument("--random-sens", dest="random_sens", action="store_true",
                   default=False,
                   help="C3 ablation: replace gate/up sensitivity with a "
                        "scale-matched random positive field.")
    p.add_argument("--grid-min", type=float, default=0.5)
    p.add_argument("--grid-max", type=float, default=1.0)
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-grid", type=int, default=20)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-tokens-per-sample", type=int, default=512)
    p.add_argument("--layer-batch-size", type=int, default=16)
    p.add_argument("--max-tok-total", type=int, default=4096)
    p.add_argument("--calib-dataset", type=str, default="c4",
                   choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", type=str, default="./calibration_cache")
    p.add_argument("--skip-lmhead", dest="skip_lmhead", action="store_true", default=True)
    p.add_argument("--quant-lmhead", dest="skip_lmhead", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("=" * 78)
    print(f"NAC PTQ | metric={args.metric} | bits={args.bits} "
          f"| group={args.group_size} | rho={args.use_rho} "
          f"| power={args.power} | random_sens={args.random_sens}")
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
