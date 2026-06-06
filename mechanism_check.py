"""
mechanism_check.py
==================
Group M of the experimental program (paper Sec. 7), run BEFORE benchmarks.

M1 (metric-prediction test)
    For each studied layer, sweep clip candidates, and for each candidate measure
        predicted_M2  = e^T M_2 e          (linear response, XX^T/N)
        predicted_Mi  = e^T M_i e          (the role's local-Jacobian metric)
        true_error    = || f(Q_c(W) X) - f(W X) ||^2   (actual post-block error)
    then correlate predicted vs. true across candidates (Spearman + Kendall).
    Paper prediction: rho(M_i, true) > rho(M_2, true), especially in gated MLP.

    "f" is the genuine block map for the role:
        gate : h = SiLU(W_g x) ⊙ u           (u from the FP up projection)
        up   : h = SiLU(g) ⊙ (W_u x)         (g from the FP gate projection)
      optionally post-multiplied by W_d (the down projection) when --through-down.
      Quantizing W_g changes only the gate branch; the up branch stays FP, and
      vice-versa — this isolates the single-projection error the metric models.

M3 (sensitivity distributions)
    Histograms / quantiles of the per-(channel,token) sensitivities:
        pointwise |sigma'(W x)|         (gate viewed in isolation)
        gate      |u * SiLU'(g)|
        up        |SiLU(g)|
    If these are near-constant the metric cannot matter; spread is what the
    reweighting exploits.

NOTE: this script runs the model and quantizer (torch). Do not invoke unless
you intend GPU work; see run_mechanism.sh for the wrapped call.
"""

from __future__ import annotations

import os
import json
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import sensitivity as sens
import metric as M
from quantizer import groupwise_rtn_symmetric, make_quant_fn
import layer_registry as reg

try:
    from scipy.stats import spearmanr, kendalltau
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------
# Rank correlation fallbacks (so scipy is optional).
# --------------------------------------------------------------------------
def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def spearman(x, y):
    if _HAVE_SCIPY:
        r, _ = spearmanr(x, y)
        return float(r)
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def kendall(x, y):
    if _HAVE_SCIPY:
        t, _ = kendalltau(x, y)
        return float(t)
    x = np.asarray(x); y = np.asarray(y)
    n = len(x); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    tot = c + d
    return float((c - d) / tot) if tot else float("nan")


# --------------------------------------------------------------------------
# Stored-X capture for a chosen set of layers.
# --------------------------------------------------------------------------
class XStore:
    def __init__(self, names, max_tokens):
        self.names = set(names)
        self.max_tokens = max_tokens
        self.store = defaultdict(list)
        self.hooks = []

    def _hook(self, name):
        def hook(_m, inp, _o):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3 and x.shape[1] > self.max_tokens:
                idx = torch.randperm(x.shape[1], device=x.device)[:self.max_tokens]
                idx = idx.sort()[0]
                x = x[:, idx, :]
            self.store[name].append(x.detach().reshape(-1, x.shape[-1]).cpu().float())
        return hook

    def register(self, model):
        for name, module in model.named_modules():
            if name in self.names:
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get(self, name, device, max_tok_total):
        if name not in self.store or not self.store[name]:
            return None
        Xt = torch.cat(self.store[name], dim=0)         # (n_tok, d_in)
        if Xt.shape[0] > max_tok_total:
            idx = torch.randperm(Xt.shape[0])[:max_tok_total]
            Xt = Xt[idx]
        return Xt.t().contiguous().to(device)           # (d_in, n_tok)


def load_calib(tokenizer, dataset, n_samples, seqlen, seed, cache_dir):
    try:
        from calibration_utils import (get_c4_calibration_data,
                                        get_wikitext2_calibration_data)
        if dataset == "c4":
            return get_c4_calibration_data(tokenizer, n_samples, seqlen, seed,
                                           cache_dir=cache_dir)
        return get_wikitext2_calibration_data(tokenizer, n_samples, seqlen,
                                              seed, cache_dir=cache_dir)
    except Exception as e:
        print(f"  calibration_utils unavailable ({str(e)[:80]}); wikitext fallback.")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 0]
    return texts[:n_samples]


def run_forward(model, tokenizer, texts, n, device):
    with torch.no_grad():
        for i, text in enumerate(texts[:n]):
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


# --------------------------------------------------------------------------
# Predicted scores: full-matrix e^T M e (NOT group-wise; M1 is about whether the
# metric ranks candidates, so we score the whole row/projection at once).
# --------------------------------------------------------------------------
def pred_M2(E, X, ntok):
    # sum_t (E x_t)^2 / N  summed over all rows -> scalar
    U = E @ X                      # (d_out, n_tok)
    return (U.pow(2).sum() / ntok).item()


def pred_Mi(E, X, s2, ntok):
    U = E @ X                      # (d_out, n_tok)
    if isinstance(s2, float):
        w = U.pow(2)
    elif s2.dim() == 1:
        w = U.pow(2) * s2.unsqueeze(0)
    else:
        w = U.pow(2) * s2          # (d_out, n_tok)
    return (w.sum() / ntok).item()


# --------------------------------------------------------------------------
# True post-block error for a role. Quantize ONLY the studied projection,
# keep siblings FP, push through the genuine block map.
# --------------------------------------------------------------------------
def true_block_error(role, Wq, Wfp_sibling_g, Wfp_sibling_u, X,
                     Wd=None, through_down=False):
    """
    role == 'gate' : Wq is the quantized gate; up branch FP from Wfp_sibling_u.
    role == 'up'   : Wq is the quantized up;   gate branch FP from Wfp_sibling_g.
    Returns ||f_q - f_fp||^2 summed over tokens (and channels).
    """
    g_fp = Wfp_sibling_g @ X                    # (d_inter, n_tok)
    u_fp = Wfp_sibling_u @ X
    h_fp = sens.silu(g_fp) * u_fp               # FP reference hidden

    if role == "gate":
        g_q = Wq @ X
        h_q = sens.silu(g_q) * u_fp
    elif role == "up":
        u_q = Wq @ X
        h_q = sens.silu(g_fp) * u_q
    else:
        raise ValueError(role)

    if through_down and Wd is not None:
        y_fp = Wd @ h_fp                        # (d_model, n_tok)
        y_q = Wd @ h_q
        return (y_q - y_fp).pow(2).sum().item()
    return (h_q - h_fp).pow(2).sum().item()


# --------------------------------------------------------------------------
# M1 for one SwiGLU block (studies both gate and up projections).
# --------------------------------------------------------------------------
@torch.no_grad()
def m1_block(parts, store, args, device):
    g_name, g_mod = parts["gate"]
    u_name, u_mod = parts["up"]
    d_name, d_mod = parts["down"]
    X = store.get(g_name, device, args.max_tok_total)
    if X is None:
        X = store.get(u_name, device, args.max_tok_total)
    if X is None:
        return None
    ntok = X.shape[1]

    Wg = g_mod.weight.data.float()
    Wu = u_mod.weight.data.float()
    Wd = d_mod.weight.data.float()
    rho = sens.downstream_gain(Wd) if args.use_rho else None
    g = Wg @ X
    u = Wu @ X

    quant = lambda W, clip: groupwise_rtn_symmetric(W, clip, args.bits, args.group_size)

    out = {}
    for role in ("gate", "up"):
        Wstudy = Wg if role == "gate" else Wu
        d_out, d_in = Wstudy.shape
        n_groups = (d_in + args.group_size - 1) // args.group_size
        pad = n_groups * args.group_size - d_in
        if pad > 0:
            Wp = torch.zeros(d_out, n_groups * args.group_size,
                             device=Wstudy.device, dtype=Wstudy.dtype)
            Wp[:, :d_in] = Wstudy
        else:
            Wp = Wstudy
        g_maxabs = Wp.reshape(d_out, n_groups, args.group_size).abs().amax(
            dim=2, keepdim=True).clamp(min=1e-8)

        if role == "gate":
            s2 = sens.GateSensitivity(g, u, rho=rho, act="silu").value()
        else:
            s2 = sens.UpSensitivity(g, rho=rho, act="silu").value()

        m2_scores, mi_scores, true_errs = [], [], []
        for gi in range(args.n_grid + 1):
            frac = args.grid_min + (args.grid_max - args.grid_min) * gi / args.n_grid
            clip = g_maxabs * frac
            Wq = quant(Wstudy, clip)
            E = Wq - Wstudy
            m2_scores.append(pred_M2(E, X, ntok))
            mi_scores.append(pred_Mi(E, X, s2, ntok))
            true_errs.append(true_block_error(
                role, Wq, Wg, Wu, X, Wd=Wd, through_down=args.through_down))

        out[role] = {
            "spearman_M2": spearman(m2_scores, true_errs),
            "spearman_Mi": spearman(mi_scores, true_errs),
            "kendall_M2": kendall(m2_scores, true_errs),
            "kendall_Mi": kendall(mi_scores, true_errs),
            "n_candidates": len(true_errs),
        }
        del s2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# M3 sensitivity distributions for one block.
# --------------------------------------------------------------------------
@torch.no_grad()
def m3_block(parts, store, args, device):
    g_name, g_mod = parts["gate"]
    u_name, u_mod = parts["up"]
    X = store.get(g_name, device, args.max_tok_total)
    if X is None:
        return None
    Wg = g_mod.weight.data.float()
    Wu = u_mod.weight.data.float()
    g = Wg @ X
    u = Wu @ X

    pointwise = sens.silu_prime(g).abs()        # |sigma'(g)|  (gate-in-isolation)
    gate_sens = (u * sens.silu_prime(g)).abs()  # |u * SiLU'(g)|
    up_sens = sens.silu(g).abs()                # |SiLU(g)|

    def stats(t):
        t = t.flatten().float()
        qs = torch.quantile(t, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0],
                                             device=t.device))
        return {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": qs[0].item(),
            "median": qs[1].item(),
            "p90": qs[2].item(),
            "p99": qs[3].item(),
            "max": qs[4].item(),
            # coefficient of variation: spread relative to mean. ~0 => metric
            # has nothing to exploit.
            "cv": (t.std() / (t.mean().abs() + 1e-12)).item(),
        }

    return {"pointwise_sigma'": stats(pointwise),
            "gate_u*sigma'": stats(gate_sens),
            "up_silu": stats(up_sens)}


# --------------------------------------------------------------------------
# Layer selection: a few decoder layers (early / mid / late) for cost.
# --------------------------------------------------------------------------
def pick_blocks(blocks, which):
    keys = sorted(k for k in blocks
                  if all(r in blocks[k] for r in ("gate", "up", "down")))
    if not keys:
        return []
    if which == "all":
        return keys
    n = len(keys)
    chosen = sorted({keys[0], keys[n // 2], keys[-1]})
    return chosen


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Group M mechanism checks (M1, M3).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", default="./mechanism_results.json")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-grid", type=int, default=20)
    p.add_argument("--grid-min", type=float, default=0.5)
    p.add_argument("--grid-max", type=float, default=1.0)
    p.add_argument("--use-rho", dest="use_rho", action="store_true", default=True)
    p.add_argument("--no-rho", dest="use_rho", action="store_false")
    p.add_argument("--through-down", action="store_true", default=False,
                   help="measure true error after W_d (else at hidden h).")
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-tokens-per-sample", type=int, default=256)
    p.add_argument("--max-tok-total", type=int, default=2048)
    p.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", default="./calibration_cache")
    p.add_argument("--blocks", default="sample", choices=["sample", "all"],
                   help="'sample' = first/mid/last decoder layer.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("=" * 78)
    print(f"Mechanism check | bits={args.bits} group={args.group_size} "
          f"rho={args.use_rho} through_down={args.through_down}")
    print("Model:", args.model_path)
    print("=" * 78)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    linear_layers = [(n, m) for n, m in model.named_modules()
                     if isinstance(m, nn.Linear)]
    blocks = reg.group_mlp_blocks(linear_layers)
    chosen = pick_blocks(blocks, args.blocks)
    print(f"Studying {len(chosen)} SwiGLU blocks: {chosen}")
    if not chosen:
        print("No complete gate/up/down blocks found — check naming.")
        return

    # Capture X only for the studied blocks' gate+up inputs.
    cap_names = []
    for k in chosen:
        cap_names.append(blocks[k]["gate"][0])
        cap_names.append(blocks[k]["up"][0])
    store = XStore(cap_names, args.max_tokens_per_sample)
    store.register(model)
    calib = load_calib(tok, args.calib_dataset, args.n_calib, args.seqlen,
                       args.seed, args.cache_dir)
    print("Running calibration forward passes ...")
    run_forward(model, tok, calib, args.n_calib, device)
    store.remove()

    results = {"config": vars(args), "M1": {}, "M3": {}}
    for k in chosen:
        tag = f"layer{k[1]}"
        print(f"\n[{tag}] M1 ...")
        m1 = m1_block(blocks[k], store, args, device)
        if m1:
            results["M1"][tag] = m1
            for role, r in m1.items():
                print(f"  {role:4s}  Spearman  M2={r['spearman_M2']:+.3f}  "
                      f"Mi={r['spearman_Mi']:+.3f}  "
                      f"(Kendall M2={r['kendall_M2']:+.3f} Mi={r['kendall_Mi']:+.3f})")
        print(f"[{tag}] M3 ...")
        m3 = m3_block(blocks[k], store, args, device)
        if m3:
            results["M3"][tag] = m3
            for kind, s in m3.items():
                print(f"  {kind:16s} cv={s['cv']:.3f} "
                      f"median={s['median']:.4g} p99={s['p99']:.4g}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate M1 verdict.
    wins = total = 0
    for tag, roles in results["M1"].items():
        for role, r in roles.items():
            total += 1
            if r["spearman_Mi"] > r["spearman_M2"]:
                wins += 1
    results["summary"] = {
        "M1_Mi_beats_M2": f"{wins}/{total}",
        "verdict": ("Mi ranks candidates better than M2 (supports paper)"
                    if total and wins > total / 2
                    else "Mi does NOT beat M2 — investigate"),
    }
    print("\n" + "=" * 78)
    print(f"M1 verdict: M_i beats M_2 in {wins}/{total} (role, layer) cases.")
    print(results["summary"]["verdict"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()