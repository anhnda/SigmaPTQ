"""
layer_registry.py
=================
Maps real HF module names (Llama-3.1, Mistral-v0.3, Qwen2.5) to roles, and
groups gate/up/down of the same decoder layer into one SwiGLU block so the
coupler can resolve them together.

Verified module naming (all three models share LlamaMLP-style naming):
    model.layers.{i}.mlp.gate_proj
    model.layers.{i}.mlp.up_proj
    model.layers.{i}.mlp.down_proj
    model.layers.{i}.self_attn.{q,k,v,o}_proj

Roles:
    "gate" / "up"   -> coupled SwiGLU metrics (need sibling response)
    "down"          -> linear-response M_2 (its input h has no further pointwise
                       nonlinearity before the residual add; identity geometry)
    "attn"          -> linear-response M_2 (paper G3: attention falls back to
                       identity geometry unless an attn Jacobian is instantiated)
    "other"         -> weight_mse / linear fallback
"""

from __future__ import annotations

import re


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_index(name: str):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def role_of(name: str) -> str:
    n = name.lower()
    if "gate_proj" in n:
        return "gate"
    if "up_proj" in n:
        return "up"
    if "down_proj" in n:
        return "down"
    if "self_attn" in n and any(p in n for p in
                                ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attn"
    return "other"


def block_key(name: str):
    """
    A SwiGLU block key groups gate/up/down of the same decoder layer.
    Returns (layer_index,) or None for non-MLP layers.
    """
    role = role_of(name)
    if role not in ("gate", "up", "down"):
        return None
    li = layer_index(name)
    return ("mlp", li) if li is not None else None


def group_mlp_blocks(linear_layers):
    """
    linear_layers : list of (name, module).
    Returns dict: block_key -> {"gate": (name,m), "up": (...), "down": (...)}.
    Only complete blocks (all three present) are usable for coupled metrics.
    """
    blocks = {}
    for name, module in linear_layers:
        key = block_key(name)
        if key is None:
            continue
        blocks.setdefault(key, {})[role_of(name)] = (name, module)
    return blocks
