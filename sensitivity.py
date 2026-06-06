"""
sensitivity.py
==============
Per-token squared sensitivities  s^2_{i,t}  from the local-Jacobian metric

    M_i = (1/N) X diag(s^2_{i,t}) X^T          (paper Eq. 8)

The ONLY thing that varies across M_2, M_sigma, M_gate, M_up is how s^2_{i,t}
is produced. Each provider returns a tensor whose SHAPE encodes its kind:

    scalar         : s2 is the python float 1.0          -> linear response M_2
    per_token      : s2 shape (n_tok,)   row-shared       (rarely used here)
    per_row_token  : s2 shape (d_out, n_tok) per (row, token)

The generic reducer in metric.py consumes any of these uniformly.

Verified architecture (Llama-3.1, Mistral-v0.3, Qwen2.5, all HF):
    h_t = SiLU(g_t) ⊙ u_t ,   g_t = W_g x_t ,  u_t = W_u x_t ,  y_t = W_d h_t

Paper sensitivities (Eqs. 11, 13, 14), per intermediate channel k:
    gate :  s^2_{k,t} = rho_k * u_{k,t}^2 * SiLU'(g_{k,t})^2
    up   :  s^2_{k,t} = rho_k * SiLU(g_{k,t})^2
    rho_k = ||W_d^{(:,k)}||^2   (downstream gain; rho_k ≡ 1 disables it)

g_{k,t} and u_{k,t} are the SIBLING projections' linear responses, so the gate
metric needs W_u@X and the up metric needs W_g@X. Neither layer can compute its
own metric in isolation; the block-level coupler (see coupling.py) supplies the
sibling response.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# SiLU and its derivative (Swish, beta = 1):  SiLU(z) = z * sigmoid(z)
#   SiLU'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
# --------------------------------------------------------------------------
def silu(z: torch.Tensor) -> torch.Tensor:
    return z * torch.sigmoid(z)


def silu_prime(z: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(z)
    return s + z * s * (1.0 - s)


def gelu_prime(z: torch.Tensor) -> torch.Tensor:
    """tanh-approx GELU derivative, for any GeGLU model (e.g. Gemma)."""
    c = 0.7978845608028654  # sqrt(2/pi)
    k = 0.044715
    inner = c * (z + k * z.pow(3))
    t = torch.tanh(inner)
    dt = (1.0 - t.pow(2)) * c * (1.0 + 3.0 * k * z.pow(2))
    return 0.5 * (1.0 + t) + 0.5 * z * dt


def gelu(z: torch.Tensor) -> torch.Tensor:
    c = 0.7978845608028654
    k = 0.044715
    return 0.5 * z * (1.0 + torch.tanh(c * (z + k * z.pow(3))))


_ACT_VALUE = {"silu": silu, "gelu": gelu}
_ACT_SLOPE = {"silu": silu_prime, "gelu": gelu_prime}


# --------------------------------------------------------------------------
# Sensitivity providers
# --------------------------------------------------------------------------
class Sensitivity:
    """Produces s^2_{i,t} for a layer. Subclasses set .kind and .value()."""

    #: one of {"scalar", "per_token", "per_row_token"}
    kind = "scalar"

    def value(self) -> object:
        """Return 1.0 (scalar), (n_tok,) tensor, or (d_out, n_tok) tensor."""
        raise NotImplementedError


class ConstantSensitivity(Sensitivity):
    """M_2 linear response:  s^2_{i,t} = 1  (Corollary 1, identity case)."""
    kind = "scalar"

    def value(self):
        return 1.0


class PointwiseSensitivity(Sensitivity):
    """
    M_sigma pointwise post-activation:  s^2_{i,t} = sigma'(w_i^T x_t)^2.
    Per-row because the pre-activation a_{i,t}=w_i^T x_t differs by row.
    For a SwiGLU gate viewed in isolation this is the OLD (incomplete) gate
    metric: it lacks the u_{k,t}^2 factor. Kept as a baseline tier.
    """
    kind = "per_row_token"

    def __init__(self, A: torch.Tensor, act: str = "silu"):
        # A = W @ X, shape (d_out, n_tok); the layer's own linear response.
        slope = _ACT_SLOPE[act.lower()]
        self._s2 = slope(A).pow(2)  # (d_out, n_tok)

    def value(self):
        return self._s2


class GateSensitivity(Sensitivity):
    """
    SwiGLU gate metric (Eq. 11, optionally Eq. 14 with rho):
        s^2_{k,t} = rho_k * u_{k,t}^2 * SiLU'(g_{k,t})^2
    Needs BOTH g = W_g X (own response) and u = W_u X (sibling response).
    """
    kind = "per_row_token"

    def __init__(self, g: torch.Tensor, u: torch.Tensor,
                 rho: torch.Tensor | None = None, act: str = "silu",
                 power: int = 2):
        # g, u : (d_inter, n_tok). rho : (d_inter,) or None.
        # power=2 -> XS^2X^T (correct, Eq.11); power=1 -> XSX^T (C1 control).
        slope = _ACT_SLOPE[act.lower()]
        base = (u.abs() * slope(g).abs())            # |s|_{k,t}, the raw magnitude
        s2 = base.pow(power)                         # |s|^power
        if rho is not None:
            # rho is a squared gain; apply rho^(power/2) so it scales like the
            # base term (rho^1 with power=2, sqrt(rho) with power=1).
            s2 = s2 * rho.pow(power / 2.0).unsqueeze(1)
        self._s2 = s2

    def value(self):
        return self._s2


class UpSensitivity(Sensitivity):
    """
    SwiGLU up metric (Eq. 13, optionally Eq. 14 with rho):
        s^2_{k,t} = rho_k * SiLU(g_{k,t})^2
    Needs the SIBLING gate response g = W_g X; the up layer never sees it
    locally. power=1 gives the XSX^T control for ablation C1.
    """
    kind = "per_row_token"

    def __init__(self, g: torch.Tensor,
                 rho: torch.Tensor | None = None, act: str = "silu",
                 power: int = 2):
        act_fn = _ACT_VALUE[act.lower()]
        base = act_fn(g).abs()                       # |SiLU(g)|_{k,t}
        s2 = base.pow(power)
        if rho is not None:
            s2 = s2 * rho.pow(power / 2.0).unsqueeze(1)
        self._s2 = s2

    def value(self):
        return self._s2


class ScaleMatchedRandomSensitivity(Sensitivity):
    """
    C3 control: replace s^2_{k,t} with RANDOM positive weights, to rule out
    "any reweighting helps." The random field is matched to a reference
    sensitivity in (a) per-row mean and (b) overall mean, so the ablation
    isolates the *structure* of the sensitivity, not its scale.

    Given a reference s2_ref of shape (d_inter, n_tok), draw r ~ Exp(1) of the
    same shape and rescale each row so its mean equals the reference row mean.
    Exp(1) gives a positive, heavy-ish-tailed field with nontrivial spread
    (so this is not "uniform = identity in disguise").
    """
    kind = "per_row_token"

    def __init__(self, s2_ref: torch.Tensor, seed: int = 0):
        gen = torch.Generator(device=s2_ref.device)
        gen.manual_seed(seed)
        # Exp(1) via inverse-CDF of U(0,1): -log(1-U).
        u = torch.rand(s2_ref.shape, generator=gen, device=s2_ref.device,
                       dtype=s2_ref.dtype).clamp(min=1e-12, max=1.0 - 1e-12)
        r = -torch.log1p(-u)                          # Exp(1), (d_inter, n_tok)
        row_mean_ref = s2_ref.mean(dim=1, keepdim=True).clamp(min=1e-20)
        row_mean_r = r.mean(dim=1, keepdim=True).clamp(min=1e-20)
        self._s2 = r * (row_mean_ref / row_mean_r)    # per-row scale match
        self.kind = "per_row_token"

    def value(self):
        return self._s2


def downstream_gain(W_down: torch.Tensor) -> torch.Tensor:
    """
    rho_k = ||W_d^{(:,k)}||^2  (Eq. 14). W_down is the down_proj weight of shape
    (d_model, d_inter); column k feeds intermediate channel k, which is exactly
    output row k of gate/up. Returns (d_inter,).
    """
    return W_down.float().pow(2).sum(dim=0)  # sum over d_model -> (d_inter,)
