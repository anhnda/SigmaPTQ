"""
collect_ppl.py
==============
Gather every ppl.json under a directory into one table and compute the paper's
verdicts (G core, C1 squared, C2r rho, C3 random). Pure stdlib; no torch.

Usage:  python collect_ppl.py ./quantized_models
"""

from __future__ import annotations

import os
import sys
import json
import glob


def load_all(root):
    rows = {}
    for path in glob.glob(os.path.join(root, "**", "ppl.json"), recursive=True):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        name = os.path.basename(os.path.dirname(path))
        rows[name] = d.get("ppl", {})
    return rows


def fmt_table(rows):
    datasets = sorted({k for v in rows.values() for k in v})
    w = max([len("checkpoint")] + [len(n) for n in rows]) + 2
    head = "checkpoint".ljust(w) + "  ".join(d.rjust(12) for d in datasets)
    lines = [head, "-" * len(head)]
    for name in sorted(rows):
        cells = "  ".join(
            (f"{rows[name][d]:.4f}".rjust(12) if d in rows[name]
             else "—".rjust(12)) for d in datasets)
        lines.append(name.ljust(w) + cells)
    return "\n".join(lines)


def _get(rows, name, ds):
    return rows.get(name, {}).get(ds)


def verdicts(rows):
    out = []

    def cmp(a, b, ds, rel="<="):
        va, vb = _get(rows, a, ds), _get(rows, b, ds)
        if va is None or vb is None:
            return None
        ok = va <= vb if rel == "<=" else va < vb
        return (ok, va, vb)

    for ds in ("wikitext2", "c4"):
        # G core: gated <= pointwise <= linear
        chain = []
        for a, b in (("g_gated", "g_pointwise"), ("g_pointwise", "g_linear")):
            r = cmp(a, b, ds)
            if r:
                chain.append((a, b, r))
        if chain:
            ok = all(r[0] for _, _, r in chain)
            parts = " <= ".join(
                f"{a}({r[1]:.3f})" for a, b, r in chain) + \
                f" <= {chain[-1][1]}({chain[-1][2][2]:.3f})"
            out.append(f"[G {ds}] {'PASS' if ok else 'FAIL'}: {parts}")

        # C1: c1c(power2) <= c1b(power1) <= c1a(linear)
        c1 = []
        for a, b in (("c1c_gated_p2_w3g128", "c1b_gated_p1_w3g128"),
                     ("c1b_gated_p1_w3g128", "c1a_linear_w3g128")):
            r = cmp(a, b, ds)
            if r:
                c1.append((a, b, r))
        if c1:
            ok = all(r[0] for _, _, r in c1)
            out.append(f"[C1 {ds}] {'PASS' if ok else 'FAIL'}: "
                       f"squared<=unsquared<=linear ({ok})")

        # C2r: no-rho primary <= +rho
        r = cmp("c1c_gated_p2_w3g128", "c2r_gated_rho_w3g128", ds)
        if r:
            out.append(f"[C2r {ds}] {'PASS' if r[0] else 'FAIL'}: "
                       f"no-rho({r[1]:.3f}) <= +rho({r[2]:.3f})")

        # C3: real gated < random
        r = cmp("c1c_gated_p2_w3g128", "c3_random_w3g128", ds, rel="<")
        if r:
            out.append(f"[C3 {ds}] {'PASS' if r[0] else 'FAIL'}: "
                       f"gated({r[1]:.3f}) < random({r[2]:.3f})")
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "./quantized_models"
    rows = load_all(root)
    if not rows:
        print(f"No ppl.json found under {root}")
        return
    print(fmt_table(rows))
    print()
    for line in verdicts(rows):
        print(line)


if __name__ == "__main__":
    main()
