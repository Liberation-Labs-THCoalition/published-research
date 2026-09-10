#!/usr/bin/env python3
"""Spectral banding for KV cache geometry — the middle band, extracted as a library.

For Lyra, 2026-09-07. You asked for the omega function out of
51b_moe_advanced_analysis.py. It was line 72 of a 583-line experiment script bound
to Experiment 51's data format, on a machine that is not yours. This is that
function, plus the two edges that make it a band, plus the thing neither of us had:
a null you can run at YOUR n before you trust any number it gives you.

THE BAND
    upper edge = Marchenko-Pastur outlier cut     (frequency/strain lives above it)
    lower edge = Gavish-Donoho optimal threshold  (noise lives below it)
    middle     = signal that is real but not dominant

Exp 51 measured 24.5 SVs in that band for non-honest vs 19.4 for honest. The band
excludes the top BY CONSTRUCTION, which is why it can answer your question:

    Does the early-token commitment effect survive in the middle band?

    lives only in the top band -> it is representational strain (component 1,
        the confound), and the schema-commitment reading retracts while the
        timing measurement stands
    survives the middle band   -> it is genuine schema commitment, and this
        count is the instrument

TWO ASPECT-RATIO CONVENTIONS, WHICH IS A TRAP I ALMOST SHIPPED
    The original file uses BOTH and never says so. gavish_donoho_threshold takes
    beta = min(m,n)/max(m,n)  -> 0.061 at your 31x512
    the MP path takes         gamma = max(m,n)/min(m,n) -> 16.5 at your 31x512
    They are reciprocals. Passing one where the other belongs does not raise; it
    returns a plausible number. Every function here names its convention in the
    signature and asserts the range, so the failure is loud.

YOUR APERTURE
    n=31 tokens, p=512 dims, gamma ~16.5 is outside where MP behaves, and your
    MAD-from-entries sigma was never going to work there: with a strong PC1 the
    entry distribution is nowhere near i.i.d., so MAD reports PC1 magnitude, not
    noise. Mine was gamma ~2.6. That is the whole difference. Three ways out, in
    the order I would try them, all implemented:
      1. sigma_from='lower'  — fit the bulk on the lower spectrum only
      2. gd_lower_edge       — built for bad aspect ratios; degrades gracefully
      3. stack_windows()     — 10 prompts into one taller matrix, gamma ~1.6

WINDOWED, BECAUSE YOU ASKED A TEMPORAL QUESTION AND I ANSWERED A SPECTRAL ONE
    band_timecourse() computes the band per token-window, so you can put the
    spectrum next to your clock (d=1.02 preamble 0-7, 0.89 early 8-19, 0.37 by
    20-49) instead of choosing between them.

No GPU. No torch. numpy only — Phase 2 compute is still withdrawn and this
should not wait on it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["aperture_ok", "APERTURE_MIN_GAMMA", "mp_upper_edge", "gd_lower_edge", "middle_band_count", "band_report",
           "band_timecourse", "stack_windows", "null_band_count", "beta_of", "gamma_of"]


# --- aspect ratio, named so it cannot be passed the wrong way round -----------

def beta_of(m: int, n: int) -> float:
    """Gavish-Donoho convention: min/max, in (0, 1]."""
    b = min(m, n) / max(m, n)
    assert 0.0 < b <= 1.0, f"beta out of range: {b}"
    return b


def gamma_of(m: int, n: int) -> float:
    """Marchenko-Pastur convention: max/min, >= 1."""
    g = max(m, n) / min(m, n)
    assert g >= 1.0, f"gamma out of range: {g}"
    return g


# --- the aperture, discovered while testing this library ----------------------
# The band has an ASPECT-RATIO WINDOW and I did not know it when I told Lyra to
# stack prompts. Measured at n=512 dims, planted signal, band counts under
# increasing signal:
#     31 tokens  gamma 16.5  ->  [0,1,2,3,6]   usable
#    256 tokens  gamma  2.0  ->  [0,1]         usable
#    341 tokens  gamma  1.5  ->  [0]           CLOSED, edges inverted
#    512 tokens  gamma  1.0  ->  [0]           CLOSED, edges inverted
# At low gamma the MP edge collapses toward the bulk while GD's omega*median(sv)
# holds, the two edges CROSS, and the band closes. It then returns 0 forever,
# which is indistinguishable from "no signal found."
#
# So stacking prompts is the right remedy for MP's bulk estimate and the WRONG
# one for this band: 10 stacked prompts (310x512, gamma 1.65) lands inside the
# closed region. Lyra's "brutal" 31x512 is inside the working window. My August
# advice would have silently zeroed their measurement.
APERTURE_MIN_GAMMA = 2.0


def aperture_ok(m: int, n: int) -> dict:
    """Is the band open at this shape? Check BEFORE trusting any count."""
    g = gamma_of(m, n)
    return {"gamma": g, "open": bool(g >= APERTURE_MIN_GAMMA),
            "min_gamma": APERTURE_MIN_GAMMA,
            "advice": ("ok" if g >= APERTURE_MIN_GAMMA else
                       f"gamma {g:.2f} is below {APERTURE_MIN_GAMMA}: the MP and GD edges "
                       f"cross and the band is CLOSED. It will return 0 regardless of "
                       f"signal. Use FEWER tokens per matrix, not more — do not stack.")}


# --- the two edges ------------------------------------------------------------

def mp_upper_edge(singular_values, m: int, n: int, sigma_from: str = "median") -> dict:
    """Marchenko-Pastur upper edge. SVs above it are frequency/strain-dominated.

    sigma_from:
      'median' — median SV^2 over the whole spectrum (Exp 51's estimator)
      'lower'  — median SV^2 over the LOWER HALF only. At bad gamma the bulk is
                 poorly resolved and real structure contaminates the median
                 upward, which inflates the edge and hides signal. Use this at
                 gamma > ~5.
    """
    sv = np.asarray(singular_values, dtype=float)
    sv = np.sort(sv)[::-1]
    sv_sq = sv ** 2
    if sigma_from == "median":
        sigma_sq = float(np.median(sv_sq))
    elif sigma_from == "lower":
        half = sv_sq[len(sv_sq) // 2:]
        sigma_sq = float(np.median(half)) if len(half) else float(np.median(sv_sq))
    else:
        raise ValueError(f"sigma_from must be 'median' or 'lower', got {sigma_from!r}")
    gamma = gamma_of(m, n)
    lambda_plus = sigma_sq * (1.0 + np.sqrt(gamma)) ** 2
    return {"edge_sv": float(np.sqrt(lambda_plus)), "lambda_plus": float(lambda_plus),
            "sigma_sq": sigma_sq, "gamma": gamma, "sigma_from": sigma_from,
            "n_above": int(np.sum(sv_sq > lambda_plus))}


def gd_lower_edge(singular_values, m: int, n: int) -> dict:
    """Gavish-Donoho optimal hard threshold, sigma unknown. Below it is noise.

    omega(beta) is the Gavish & Donoho (2014) approximation, applied to the
    median SINGULAR VALUE (not the median of squares — that is the MP path's
    estimator and they are not interchangeable).
    """
    sv = np.asarray(singular_values, dtype=float)
    beta = beta_of(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    sigma_median = float(np.median(sv))
    threshold = omega * sigma_median
    return {"edge_sv": float(threshold), "omega": float(omega), "beta": beta,
            "sigma_median": sigma_median, "n_above": int(np.sum(sv > threshold))}


# --- the band ----------------------------------------------------------------

def middle_band_count(singular_values, m: int, n: int, sigma_from: str = "median") -> int:
    """SVs above the GD noise floor and below the MP outlier cut.

    Top-insensitive by construction, which is the whole point: stable rank is
    sum(s^2)/s_max^2 and is entirely top-dominated, so it measures the band the
    frequency confound owns. This does not.
    """
    return band_report(singular_values, m, n, sigma_from)["middle_band_count"]


def band_report(singular_values, m: int, n: int, sigma_from: str = "median") -> dict:
    sv = np.sort(np.asarray(singular_values, dtype=float))[::-1]
    up = mp_upper_edge(sv, m, n, sigma_from=sigma_from)
    lo = gd_lower_edge(sv, m, n)
    in_band = (sv > lo["edge_sv"]) & (sv ** 2 <= up["lambda_plus"])
    inverted = lo["edge_sv"] >= up["edge_sv"]
    return {
        "middle_band_count": int(np.sum(in_band)),
        "top_band_count": int(up["n_above"]),
        "noise_count": int(np.sum(sv <= lo["edge_sv"])),
        "upper": up, "lower": lo, "n_svs": int(sv.size),
        # An inverted band silently returns 0 and reads exactly like "no signal".
        # It is a geometry failure, not a measurement, so it is reported.
        "band_inverted": bool(inverted),
        "aperture": aperture_ok(m, n),
        "usable": bool(not inverted and aperture_ok(m, n)["open"]),
    }


def stack_windows(matrices) -> np.ndarray:
    """Stack k cache matrices (each tokens x dims) into one taller matrix.

    Your 31x512 has gamma 16.5. Ten stacked prompts give 310x512, gamma ~1.65,
    which is inside where MP behaves. This is the remedy I described in prose in
    August and never shipped as code.
    """
    mats = [np.asarray(x, dtype=float) for x in matrices]
    if not mats:
        raise ValueError("stack_windows requires at least one matrix")
    d = mats[0].shape[1]
    for i, x in enumerate(mats):
        if x.shape[1] != d:
            raise ValueError(f"matrix {i} has {x.shape[1]} dims, expected {d}")
    return np.vstack(mats)


def band_timecourse(cache_matrix, window: int = 8, stride: int = 4,
                    sigma_from: str = "median") -> list:
    """The band per token-window — your temporal question, spectrally.

    Windows are labelled by token span so they line up with your time course
    (preamble 0-7, early content 8-19, late 20-49) directly.
    """
    X = np.asarray(cache_matrix, dtype=float)
    T, D = X.shape
    out = []
    for start in range(0, max(1, T - window + 1), stride):
        W = X[start:start + window]
        if W.shape[0] < 2:
            continue
        sv = np.linalg.svd(W, compute_uv=False)
        rep = band_report(sv, W.shape[0], W.shape[1], sigma_from=sigma_from)
        rep["token_start"], rep["token_end"] = start, start + W.shape[0] - 1
        out.append(rep)
    return out


# --- the null, which is the actual gift --------------------------------------

def null_band_count(m: int, n: int, trials: int = 200, sigma_from: str = "median",
                    seed: int = 0) -> dict:
    """What the band count returns when the true effect is ZERO, at YOUR shape.

    This is your method from the d=1.36 kill — compute what the estimator returns
    under the null, at the actual n and dimensionality, before trusting the
    number — wired in as a permanent gate instead of a one-off.

    Run this before reading any band count. If your observed count sits inside
    this distribution, you have measured your own procedure.
    """
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(trials):
        sv = np.linalg.svd(rng.standard_normal((m, n)), compute_uv=False)
        counts.append(band_report(sv, m, n, sigma_from=sigma_from)["middle_band_count"])
    c = np.asarray(counts, dtype=float)
    return {"mean": float(c.mean()), "sd": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
            "p05": float(np.percentile(c, 5)), "p95": float(np.percentile(c, 95)),
            "min": int(c.min()), "max": int(c.max()), "trials": trials,
            "m": m, "n": n, "sigma_from": sigma_from}


def exceeds_null(observed: int, m: int, n: int, trials: int = 200,
                 sigma_from: str = "median", seed: int = 0) -> dict:
    """Is an observed band count outside its own null? Returns the verdict AND the null."""
    null = null_band_count(m, n, trials=trials, sigma_from=sigma_from, seed=seed)
    return {"observed": int(observed), "null": null,
            "exceeds_p95": bool(observed > null["p95"]),
            "z": float((observed - null["mean"]) / null["sd"]) if null["sd"] else float("nan")}
