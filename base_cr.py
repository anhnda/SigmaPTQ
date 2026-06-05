"""
base_cr.py
==========
Clip-range selection interface for weight-only group-wise PTQ.

Clip selection is cast as METRIC SELECTION (Sigma-Aware PTQ paper, Eq. 5):

    c* = argmin_c  e(c)^T M e(c),     e(c) = Q_c(w) - w

The entire design space is the PSD metric M. This module provides:

  Tier 1  WeightMSE       M = I                       (penalize ||e||^2)
  Tier 2  LinearResponse  M = (1/N) X X^T             (penalize delta of W x)
  Tier 3  Mixed           M = (1-lam) I + lam (XX^T)  (weight-space floor)
  Paper   SigmaAware      M = (1/N) X S^2 X^T         (penalize delta of sigma(Wx))
          FlooredSigma    M = (1-lam) I + lam (X S^2 X^T)

Design:
  - Each clip-range method is a ClipRange subclass exposing build_metric()
    (returns the per-row metric machinery) and select_clip().
  - select_clip() runs the standard grid search over fractions of the
    per-row max-abs value of w, scoring each candidate by the quadratic form.
  - The quantizer (RTN, group-wise) is supplied as a callable so the same
    clip search works for any quantization grid.

All metrics act on the d_in x d_in input-feature space. For a weight row
w in R^{d_in}, the error e in R^{d_in} is scored by e^T M e. Because M can
be large (d_in x d_in), we never materialize it when only the quadratic
form is needed: e^T (X X^T) e = ||X^T e||^2 is computed via matvecs.
"""

from __future__ import annotations

import torch
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Quantizer signature shared by all clip-range methods.
#   quant_fn(W_row_or_block, clip) -> dequantized tensor of same shape
# The clip-range layer only needs the *error* e = Q_c(w) - w, so it calls
# quant_fn and subtracts. quant_fn encapsulates group-wise RTN.
# --------------------------------------------------------------------------
QuantFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ClipRange:
    """
    Base clip-range selector.

    Subclasses implement `score(e)` returning the per-row scalar (or vector
    over rows) quadratic form e^T M e. The grid search machinery in
    `select_clip` is shared.

    Parameters
    ----------
    n_grid : int
        Number of grid points in the clip-fraction search.
    grid_min, grid_max : float
        Search clip = frac * max_abs(w) for frac in [grid_min, grid_max].
    """

    name = "base"

    def __init__(self, n_grid: int = 20, grid_min: float = 0.5, grid_max: float = 1.0):
        self.n_grid = n_grid
        self.grid_min = grid_min
        self.grid_max = grid_max

    # ---- to be provided by subclasses --------------------------------------
    def prepare(self, X: Optional[torch.Tensor], W: torch.Tensor, pre_act_fn=None):
        """
        Cache whatever the metric needs (e.g. X, or X*S). Called once per
        layer before the per-row/per-block search.

        X : (d_in, n_tok) calibration activations for this layer, or None.
        W : (d_out, d_in) weight.
        pre_act_fn : optional callable mapping pre-activations A=W@X to slopes S.
        """
        self.X = X
        self.W = W

    def score(self, E: torch.Tensor) -> torch.Tensor:
        """
        Quadratic form e^T M e for each row.

        E : (d_out, d_in) error matrix (one row per output channel).
        returns : (d_out,) score per row.
        """
        raise NotImplementedError

    # ---- shared grid search ------------------------------------------------
    @torch.no_grad()
    def select_clip(self, W: torch.Tensor, quant_fn: QuantFn) -> torch.Tensor:
        """
        Per-row clip selection via grid search.

        W : (d_out, d_in)
        quant_fn(W, clip_per_row) -> dequantized W of same shape, where
            clip_per_row is (d_out, 1) broadcastable.

        returns : best clip per row, shape (d_out, 1).
        """
        d_out = W.shape[0]
        max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (d_out,1)

        best_score = torch.full((d_out,), float("inf"), device=W.device)
        best_clip = max_abs.clone()

        for g in range(self.n_grid + 1):
            frac = self.grid_min + (self.grid_max - self.grid_min) * g / self.n_grid
            clip = max_abs * frac                       # (d_out,1)
            Wq = quant_fn(W, clip)                       # (d_out,d_in)
            E = Wq - W
            s = self.score(E)                            # (d_out,)
            improve = s < best_score
            best_score = torch.where(improve, s, best_score)
            best_clip = torch.where(improve.unsqueeze(1), clip, best_clip)

        return best_clip


# ==========================================================================
# Tier 1 : weight-MSE   M = I
# ==========================================================================
class WeightMSE(ClipRange):
    name = "weight_mse"

    def prepare(self, X, W, pre_act_fn=None):
        self.W = W  # X unused

    def score(self, E: torch.Tensor) -> torch.Tensor:
        # e^T I e = ||e||^2 per row
        return E.pow(2).sum(dim=1)


# ==========================================================================
# Tier 2 : linear-response   M = (1/N) X X^T
#   e^T M e = (1/N) ||X^T e||^2
# ==========================================================================
class LinearResponse(ClipRange):
    name = "linear_response"

    def prepare(self, X, W, pre_act_fn=None):
        assert X is not None, "LinearResponse needs calibration activations X"
        # X : (d_in, n_tok). Store as is; score uses matvec to avoid d_in x d_in.
        self.X = X
        self.ntok = X.shape[1]

    def score(self, E: torch.Tensor) -> torch.Tensor:
        # E : (d_out, d_in);  X : (d_in, n_tok)
        # per row e_r:  e_r^T X X^T e_r = || E X ||^2 row-wise / N
        EX = E @ self.X                       # (d_out, n_tok)
        return EX.pow(2).sum(dim=1) / self.ntok


# ==========================================================================
# Tier 3 : mixed  M = (1-lam) I + lam M_inner
#   inner metric defaults to linear-response XX^T, but can be sigma-aware.
# ==========================================================================
class Mixed(ClipRange):
    name = "mixed"

    def __init__(self, lam: float = 0.5, inner: str = "linear",
                 n_grid: int = 20, grid_min: float = 0.5, grid_max: float = 1.0):
        super().__init__(n_grid, grid_min, grid_max)
        self.lam = float(lam)
        self.inner = inner  # "linear" -> XX^T,  "sigma" -> X S^2 X^T

    def prepare(self, X, W, pre_act_fn=None):
        assert X is not None, "Mixed needs calibration activations X"
        self.W = W
        if self.inner == "sigma":
            assert pre_act_fn is not None, "sigma inner metric needs pre_act_fn"
            A = W @ X                      # (d_out, n_tok) pre-activations
            S = pre_act_fn(A)              # (d_out, n_tok) slopes sigma'(A)
            # Mixed sigma uses a single shared S per layer; use mean over rows
            # to keep one X-side weighting (per-row sigma handled in SigmaAware).
            s = S.mean(dim=0).clamp(min=0.0)        # (n_tok,)
            self.XS = X * s.unsqueeze(0)            # (d_in, n_tok)
            self.ntok = X.shape[1]
        else:
            self.X = X
            self.ntok = X.shape[1]

    def score(self, E: torch.Tensor) -> torch.Tensor:
        weight_term = E.pow(2).sum(dim=1)                      # (1-lam) part base
        if self.inner == "sigma":
            EX = E @ self.XS
        else:
            EX = E @ self.X
        inner_term = EX.pow(2).sum(dim=1) / self.ntok
        return (1.0 - self.lam) * weight_term + self.lam * inner_term


# ==========================================================================
# Paper method : sigma-aware   M_sigma = (1/N) X S^2 X^T
#   Optional identity floor -> Floored-NAC: (1-lam) I + lam M_sigma
#   Slopes are per output-row (S = sigma'(W x) depends on the row's pre-act),
#   so the metric is genuinely per-row.
# ==========================================================================
class SigmaAware(ClipRange):
    name = "sigma_aware"

    def __init__(self, lam: float = 0.0, n_grid: int = 20,
                 grid_min: float = 0.5, grid_max: float = 1.0):
        # lam = 0 -> unfloored NAC ; lam in (0,1) -> Floored-NAC
        super().__init__(n_grid, grid_min, grid_max)
        self.lam = float(lam)

    def prepare(self, X, W, pre_act_fn=None):
        assert X is not None, "SigmaAware needs calibration activations X"
        assert pre_act_fn is not None, "SigmaAware needs pre_act_fn for slopes"
        self.X = X                           # (d_in, n_tok)
        A = W @ X                            # (d_out, n_tok) pre-activations
        self.S = pre_act_fn(A)               # (d_out, n_tok) slopes sigma'(A)
        self.ntok = X.shape[1]

    def score(self, E: torch.Tensor) -> torch.Tensor:
        # Per row r:  e_r^T (1/N X diag(s_r^2) X^T) e_r
        #           = (1/N) sum_t s_{r,t}^2 (e_r . x_t)^2
        # e_r . x_t  ->  (E @ X)[r, t] = u_{r,t}
        U = E @ self.X                        # (d_out, n_tok)
        sigma_term = (self.S.pow(2) * U.pow(2)).sum(dim=1) / self.ntok
        if self.lam > 0.0:
            weight_term = E.pow(2).sum(dim=1)
            return (1.0 - self.lam) * weight_term + self.lam * sigma_term
        return sigma_term


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_clip_range(kind: str, lam: float = 0.5, inner: str = "linear",
                     n_grid: int = 20, grid_min: float = 0.5,
                     grid_max: float = 1.0) -> ClipRange:
    kind = kind.lower()
    if kind == "weight_mse":
        return WeightMSE(n_grid, grid_min, grid_max)
    if kind == "linear_response":
        return LinearResponse(n_grid, grid_min, grid_max)
    if kind == "mixed":
        return Mixed(lam=lam, inner=inner, n_grid=n_grid,
                     grid_min=grid_min, grid_max=grid_max)
    if kind == "sigma_aware":
        return SigmaAware(lam=lam, n_grid=n_grid, grid_min=grid_min, grid_max=grid_max)
    raise ValueError(f"Unknown clip-range kind: {kind!r}. "
                     "Choose from weight_mse | linear_response | mixed | sigma_aware")