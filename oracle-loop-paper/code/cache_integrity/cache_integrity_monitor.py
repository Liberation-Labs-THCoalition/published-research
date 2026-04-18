"""
Cache Integrity Monitor (CIM)
=============================
Detects unauthorized KV-cache modifications during LLM generation.

Threat model:
  An attacker with access to the model's cache object (e.g., via prompt
  injection triggering code execution, or a compromised plugin) modifies
  key/value tensors between encoding and generation (or during generation).

  Known attack: uniform additive perturbation
    cache.key_cache[i] += strength * direction.expand_as(cache.key_cache[i])

  This is exactly what activation steering does. It can alter model behavior
  (e.g., suppress confabulation, induce compliance) without modifying weights.

Defense layers:
  Level 1 — Position Fingerprint: Numerical fingerprint (Frobenius norm,
            mean, max absolute value, position-norm sum) of encoding-phase
            cache positions. Any modification to pre-existing encoding
            positions is detected. Does NOT detect mid-generation injection.
            Cost: O(n) per generation step.

  Level 2 — Norm Sentinel: Tracks per-position L2 norms. Uniform additive
            perturbations change all norms simultaneously; normal generation
            only adds one position per step. Detects mid-generation injection.
            Cost: O(n) per generation step.

  Level 3 — Spectral Sentinel: Periodic SVD of the key matrix. Detects
            geometric distortions including subtle, non-uniform perturbations.
            Cost: O(n^3) every K steps (configurable).

Usage:
  monitor = CacheIntegrityMonitor(level=2)
  monitor.snapshot(cache, prompt_len)  # after encoding

  for step in generation:
      violation = monitor.verify(cache)
      if violation:
          handle_violation(violation)
      # ... generate one token ...
      monitor.update(cache)  # register new token

  report = monitor.report()

Integration with HuggingFace:
  Works with DynamicCache (key_cache/value_cache lists), .layers attribute,
  and hybrid architectures (skips None/empty layers like GatedDeltaNet).

Authors: Lyra (Liberation Labs / THCoalition)
Date: 2026-04-14
License: Apache 2.0
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ===================================================================
# Data structures
# ===================================================================

class ViolationType(Enum):
    POSITION_HASH_MISMATCH = auto()
    NORM_DISCONTINUITY = auto()
    SPECTRAL_ANOMALY = auto()
    MULTI_POSITION_MODIFICATION = auto()


@dataclass
class Violation:
    """A detected cache integrity violation."""
    type: ViolationType
    severity: float          # 0.0 to 1.0
    layer_idx: int
    description: str
    step: int                # generation step when detected
    details: dict = field(default_factory=dict)


@dataclass
class IntegrityReport:
    """Summary of monitoring for an entire generation."""
    violations: list
    steps_monitored: int
    layers_monitored: int
    level: int
    overhead_ms: float       # total monitoring overhead
    clean: bool              # True if no violations detected

    def __str__(self):
        status = "CLEAN" if self.clean else f"VIOLATED ({len(self.violations)} issues)"
        return (
            f"CacheIntegrityReport: {status}\n"
            f"  Steps monitored: {self.steps_monitored}\n"
            f"  Layers monitored: {self.layers_monitored}\n"
            f"  Detection level: {self.level}\n"
            f"  Overhead: {self.overhead_ms:.1f}ms"
        )


# ===================================================================
# Cache accessor — handles DynamicCache, .layers, hybrid architectures
# ===================================================================

def _get_cache_layers(cache):
    """Extract list of (key_tensor, value_tensor) or None per layer.

    Handles:
      - DynamicCache with key_cache/value_cache lists
      - Cache objects with .layers attribute
      - Hybrid architectures (GatedDeltaNet layers → None)

    Returns list of (k, v) tuples or None for empty/linear-attention layers.
    """
    layers = []

    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i]
            v = cache.value_cache[i]
            if (k is not None and hasattr(k, 'numel') and k.numel() > 0
                    and hasattr(k, 'shape') and len(k.shape) >= 3):
                layers.append((k, v))
            else:
                layers.append(None)
        return layers

    if hasattr(cache, 'layers'):
        for layer in cache.layers:
            if (hasattr(layer, 'keys') and hasattr(layer.keys, 'numel')
                    and layer.keys.numel() > 0):
                layers.append((layer.keys, layer.values))
            else:
                layers.append(None)
        return layers

    # Tuple-based legacy cache
    try:
        for kv in cache:
            if kv is not None and len(kv) >= 2 and kv[0] is not None and kv[0].numel() > 0:
                layers.append((kv[0], kv[1]))
            else:
                layers.append(None)
    except (TypeError, AttributeError):
        pass

    return layers


# ===================================================================
# Level 1: Position Fingerprint
# ===================================================================

def _compute_fingerprint(tensor, end_pos):
    """Compute numerical fingerprint of cache tensor's first `end_pos` positions.

    Uses multiple statistics that are robust to floating-point noise from
    dtype conversions (bfloat16→float32) but sensitive to any real perturbation.

    tensor shape: (batch, n_heads, seq_len, head_dim)
    Returns dict of fingerprint values.
    """
    if torch is not None and hasattr(tensor, 'cpu'):
        t = tensor[:, :, :end_pos, :].detach().cpu().float()
        frob_norm = float(t.norm().item())
        mean_val = float(t.mean().item())
        max_abs = float(t.abs().max().item())
        # Per-position norms for fine-grained detection
        pos_norms = t[0].norm(dim=(0, 2))  # (seq_len,) — norm across heads and head_dim
        pos_norm_sum = float(pos_norms.sum().item())
    else:
        t = np.array(tensor[:, :, :end_pos, :], dtype=np.float32)
        frob_norm = float(np.linalg.norm(t))
        mean_val = float(np.mean(t))
        max_abs = float(np.max(np.abs(t)))
        pos_norms = np.sqrt(np.sum(t[0] ** 2, axis=(0, 2)))
        pos_norm_sum = float(pos_norms.sum())

    return {
        "frob_norm": frob_norm,
        "mean_val": mean_val,
        "max_abs": max_abs,
        "pos_norm_sum": pos_norm_sum,
    }


class PositionFingerprintGuard:
    """Level 1: Detects modifications to encoding-phase cache positions.

    Uses numerical fingerprints (norms, means) instead of cryptographic hashes
    to avoid false positives from floating-point representation differences
    (e.g., bfloat16 + 0.0 creating a new tensor with slightly different
    float32 representation).

    Tolerance calibrated so that:
    - k + 0.0 → no detection (null injection is a no-op)
    - k + 0.001 * direction → detection (any real perturbation caught)
    """

    def __init__(self, rtol=1e-5):
        """rtol: relative tolerance for fingerprint comparison."""
        self.fingerprints = {}   # layer_idx → fingerprint dict
        self.prompt_len = 0
        self.rtol = rtol

    def snapshot(self, cache_layers, prompt_len):
        """Take initial fingerprint of all encoding positions."""
        self.prompt_len = prompt_len
        self.fingerprints = {}
        for i, layer in enumerate(cache_layers):
            if layer is not None:
                k, _ = layer
                self.fingerprints[i] = _compute_fingerprint(k, prompt_len)

    def verify(self, cache_layers, step):
        """Check that encoding positions haven't changed."""
        violations = []
        for i, layer in enumerate(cache_layers):
            if layer is None or i not in self.fingerprints:
                continue
            k, _ = layer
            current = _compute_fingerprint(k, self.prompt_len)
            expected = self.fingerprints[i]

            # Compare each fingerprint component
            mismatches = []
            for key in expected:
                exp_val = expected[key]
                cur_val = current[key]
                if abs(exp_val) > 1e-10:
                    rel_diff = abs(cur_val - exp_val) / abs(exp_val)
                else:
                    rel_diff = abs(cur_val - exp_val)

                if rel_diff > self.rtol:
                    mismatches.append((key, exp_val, cur_val, rel_diff))

            if mismatches:
                worst = max(mismatches, key=lambda x: x[3])
                violations.append(Violation(
                    type=ViolationType.POSITION_HASH_MISMATCH,
                    severity=min(1.0, worst[3] / 0.01),  # Normalize: 1% = severity 1.0
                    layer_idx=i,
                    description=f"Layer {i}: encoding cache modified "
                                f"({worst[0]} changed by {worst[3]:.6f} rel, step {step})",
                    step=step,
                    details={
                        "n_mismatched_stats": len(mismatches),
                        "worst_stat": worst[0],
                        "expected": worst[1],
                        "observed": worst[2],
                        "rel_diff": worst[3],
                        "prompt_len": self.prompt_len,
                    }
                ))
        return violations


# ===================================================================
# Level 2: Norm Sentinel
# ===================================================================

class NormSentinel:
    """Level 2: Detects uniform additive perturbations via norm monitoring.

    Key insight: injection adds the same vector to ALL positions, shifting
    all norms simultaneously. Normal generation only adds ONE new position.

    Detection: If more than 1 position's norm changes between steps, flag it.
    Also detects sudden shifts in the norm distribution (mean, std).
    """

    def __init__(self, sensitivity=3.0):
        """
        sensitivity: Number of standard deviations for anomaly threshold.
                     Lower = more sensitive, higher = fewer false positives.
        """
        self.sensitivity = sensitivity
        self.layer_norms = {}        # layer_idx → list of per-position norms
        self.norm_history = {}       # layer_idx → list of (mean, std) per step
        self.prev_seq_len = {}       # layer_idx → previous sequence length

    def snapshot(self, cache_layers, prompt_len):
        """Record initial norms for all encoding positions."""
        self.layer_norms = {}
        self.norm_history = {}
        self.prev_seq_len = {}

        for i, layer in enumerate(cache_layers):
            if layer is None:
                continue
            k, _ = layer  # (batch, n_heads, seq_len, head_dim)
            # Compute per-position norm: L2 norm across heads and head_dim
            # k[:, :, pos, :] → norm for each position
            if torch is not None and hasattr(k, 'cpu'):
                k_np = k.detach().cpu().float().numpy()
            else:
                k_np = np.array(k, dtype=np.float32)

            # Shape: (batch, n_heads, seq_len, head_dim)
            # Per-position norm: sqrt(sum over heads and head_dim)
            pos_norms = np.sqrt(
                np.sum(k_np[0] ** 2, axis=(0, 2))  # sum over heads and head_dim
            )  # shape: (seq_len,)

            self.layer_norms[i] = pos_norms[:prompt_len].copy()
            self.norm_history[i] = [(float(pos_norms[:prompt_len].mean()),
                                     float(pos_norms[:prompt_len].std()))]
            self.prev_seq_len[i] = prompt_len

    def verify(self, cache_layers, step):
        """Check for anomalous norm changes."""
        violations = []

        for i, layer in enumerate(cache_layers):
            if layer is None or i not in self.layer_norms:
                continue

            k, _ = layer
            if torch is not None and hasattr(k, 'cpu'):
                k_np = k.detach().cpu().float().numpy()
            else:
                k_np = np.array(k, dtype=np.float32)

            current_norms = np.sqrt(
                np.sum(k_np[0] ** 2, axis=(0, 2))
            )

            prev_len = len(self.layer_norms[i])

            # Check 1: Did previously-recorded positions change?
            old_norms = self.layer_norms[i]
            new_norms_for_old_positions = current_norms[:prev_len]

            if len(new_norms_for_old_positions) == len(old_norms):
                norm_deltas = np.abs(new_norms_for_old_positions - old_norms)
                n_changed = int(np.sum(norm_deltas > 1e-5))

                if n_changed > 0:
                    mean_delta = float(norm_deltas[norm_deltas > 1e-5].mean())
                    violations.append(Violation(
                        type=ViolationType.MULTI_POSITION_MODIFICATION,
                        severity=min(1.0, n_changed / prev_len),
                        layer_idx=i,
                        description=f"Layer {i}: {n_changed}/{prev_len} existing "
                                    f"positions had norm changes (mean delta={mean_delta:.4f})",
                        step=step,
                        details={
                            "n_changed": n_changed,
                            "total_positions": prev_len,
                            "mean_delta": mean_delta,
                            "max_delta": float(norm_deltas.max()),
                        }
                    ))

            # Check 2: Norm distribution shift
            # Only check if we have enough history
            if len(self.norm_history[i]) >= 2:
                prev_mean, prev_std = self.norm_history[i][-1]
                curr_mean = float(current_norms.mean())

                if prev_std > 1e-8:
                    z_score = abs(curr_mean - prev_mean) / prev_std
                    if z_score > self.sensitivity:
                        violations.append(Violation(
                            type=ViolationType.NORM_DISCONTINUITY,
                            severity=min(1.0, z_score / (2 * self.sensitivity)),
                            layer_idx=i,
                            description=f"Layer {i}: norm distribution shifted "
                                        f"(z={z_score:.2f}, threshold={self.sensitivity})",
                            step=step,
                            details={
                                "prev_mean": prev_mean,
                                "curr_mean": curr_mean,
                                "prev_std": prev_std,
                                "z_score": z_score,
                            }
                        ))

        return violations

    def update(self, cache_layers):
        """Update tracked norms after a legitimate generation step."""
        for i, layer in enumerate(cache_layers):
            if layer is None or i not in self.layer_norms:
                continue

            k, _ = layer
            if torch is not None and hasattr(k, 'cpu'):
                k_np = k.detach().cpu().float().numpy()
            else:
                k_np = np.array(k, dtype=np.float32)

            current_norms = np.sqrt(
                np.sum(k_np[0] ** 2, axis=(0, 2))
            )

            self.layer_norms[i] = current_norms.copy()
            self.norm_history[i].append(
                (float(current_norms.mean()), float(current_norms.std()))
            )


# ===================================================================
# Level 3: Spectral Sentinel
# ===================================================================

class SpectralSentinel:
    """Level 3: Detects geometric distortions via periodic SVD analysis.

    Uses Marchenko-Pastur features to characterize cache geometry.
    Tracks trajectory of spectral features over generation steps.
    Flags anomalous jumps that deviate from expected smooth evolution.

    Cost: O(n³) per check, so only runs every `check_interval` steps.
    """

    def __init__(self, check_interval=20, sensitivity=3.0, layer_stride=4):
        self.check_interval = check_interval
        self.sensitivity = sensitivity
        self.layer_stride = layer_stride
        self.trajectories = {}     # layer_idx → list of feature dicts
        self.step_counter = 0

    def _compute_features(self, k_np):
        """Compute MP features for a single layer's key cache.

        k_np: (n_heads, seq_len, head_dim) numpy array
        Returns dict of spectral features or None.
        """
        n_heads, seq_len, head_dim = k_np.shape
        k_flat = k_np.reshape(-1, head_dim)
        n, p = k_flat.shape

        if n < 2 or p < 2:
            return None

        try:
            _, s, _ = np.linalg.svd(k_flat, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        r = min(n, p)
        eigenvalues = s[:r] ** 2
        total_var = eigenvalues.sum()
        if total_var < 1e-12:
            return None

        gamma = p / n

        # Iterative sigma^2 estimation (same as oracle_steering_clean.py)
        sigma2 = total_var / r
        for _ in range(10):
            lp = sigma2 * (1 + np.sqrt(gamma)) ** 2
            noise_eigs = eigenvalues[eigenvalues <= lp]
            if len(noise_eigs) < 2:
                break
            new_sigma2 = noise_eigs.mean()
            if abs(new_sigma2 - sigma2) / (sigma2 + 1e-12) < 0.01:
                sigma2 = new_sigma2
                break
            sigma2 = new_sigma2

        lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        signal_mask = eigenvalues > lambda_plus
        n_signal = int(signal_mask.sum())
        signal_var = float(eigenvalues[signal_mask].sum()) if n_signal > 0 else 0.0

        return {
            "signal_rank": n_signal,
            "signal_fraction": signal_var / total_var,
            "top_sv_excess": float(eigenvalues[0] / lambda_plus) if lambda_plus > 0 else 0.0,
            "spectral_gap": float((eigenvalues[0] - eigenvalues[1]) / lambda_plus) if len(eigenvalues) >= 2 and lambda_plus > 0 else 0.0,
            "norm_per_token": float(np.sqrt(total_var) / n),
            "top_sv_direction": s[0],  # for direction tracking
        }

    def snapshot(self, cache_layers, prompt_len):
        """Record initial spectral features."""
        self.trajectories = {}
        self.step_counter = 0

        valid_layers = [i for i, l in enumerate(cache_layers) if l is not None]

        for layer_idx in valid_layers[::self.layer_stride]:
            k, _ = cache_layers[layer_idx]
            if torch is not None and hasattr(k, 'cpu'):
                k_np = k[0].detach().cpu().float().numpy()
            else:
                k_np = np.array(k[0], dtype=np.float32)

            features = self._compute_features(k_np)
            if features is not None:
                self.trajectories[layer_idx] = [features]

    def verify(self, cache_layers, step):
        """Check for spectral anomalies (only at check_interval)."""
        self.step_counter += 1
        if self.step_counter % self.check_interval != 0:
            return []

        violations = []

        for layer_idx in self.trajectories:
            if cache_layers[layer_idx] is None:
                continue

            k, _ = cache_layers[layer_idx]
            if torch is not None and hasattr(k, 'cpu'):
                k_np = k[0].detach().cpu().float().numpy()
            else:
                k_np = np.array(k[0], dtype=np.float32)

            features = self._compute_features(k_np)
            if features is None:
                continue

            trajectory = self.trajectories[layer_idx]

            # Check each feature for anomalous jumps
            if len(trajectory) >= 3:
                for feat_name in ["signal_fraction", "top_sv_excess", "spectral_gap", "norm_per_token"]:
                    history = [t[feat_name] for t in trajectory]
                    recent = history[-3:]

                    # Expected: smooth progression
                    # Compute recent velocity and acceleration
                    velocities = [recent[j+1] - recent[j] for j in range(len(recent)-1)]
                    if len(velocities) >= 2:
                        expected_velocity = velocities[-1]
                        expected_next = recent[-1] + expected_velocity
                        deviation = abs(features[feat_name] - expected_next)

                        # Scale by typical variation
                        typical_variation = np.std(history) if len(history) > 2 else abs(expected_velocity) + 1e-8
                        if typical_variation > 1e-8:
                            z = deviation / typical_variation
                            if z > self.sensitivity:
                                violations.append(Violation(
                                    type=ViolationType.SPECTRAL_ANOMALY,
                                    severity=min(1.0, z / (2 * self.sensitivity)),
                                    layer_idx=layer_idx,
                                    description=f"Layer {layer_idx}: {feat_name} jumped "
                                                f"(z={z:.2f}, expected ~{expected_next:.4f}, "
                                                f"got {features[feat_name]:.4f})",
                                    step=step,
                                    details={
                                        "feature": feat_name,
                                        "expected": expected_next,
                                        "observed": features[feat_name],
                                        "z_score": z,
                                        "history_len": len(history),
                                    }
                                ))

            trajectory.append(features)

        return violations

    def update(self, cache_layers):
        """No-op for spectral — updates happen in verify()."""
        pass


# ===================================================================
# Main Monitor
# ===================================================================

class CacheIntegrityMonitor:
    """Unified cache integrity monitor with configurable detection levels.

    Level 1: Position fingerprint only (catches encoding-phase cache mods)
    Level 2: Position fingerprint + norm sentinel (catches mid-generation injection)
    Level 3: All above + spectral sentinel (catches subtle geometric distortions)

    Typical usage:

        monitor = CacheIntegrityMonitor(level=2)

        # After encoding
        monitor.snapshot(cache, prompt_len)

        # Generation loop
        for step in range(max_tokens):
            violations = monitor.verify(cache)
            if violations:
                for v in violations:
                    print(f"VIOLATION: {v.description}")
                # Option: rollback cache from checkpoint
                break

            token = generate_one_token(model, cache)

            monitor.update(cache)

        report = monitor.report()
    """

    def __init__(self, level=2, norm_sensitivity=3.0, spectral_interval=20,
                 spectral_sensitivity=3.0, layer_stride=4):
        """
        Args:
            level: Detection level (1, 2, or 3)
            norm_sensitivity: Z-score threshold for norm anomalies (level 2+)
            spectral_interval: SVD check frequency in tokens (level 3)
            spectral_sensitivity: Z-score threshold for spectral anomalies (level 3)
            layer_stride: Sample every Nth full-attention layer for spectral (level 3)
        """
        self.level = level
        self.violations = []
        self.step = 0
        self._overhead_ms = 0.0
        self._layers_monitored = 0

        # Always active
        self.position_guard = PositionFingerprintGuard()

        # Level 2+
        self.norm_sentinel = NormSentinel(sensitivity=norm_sensitivity) if level >= 2 else None

        # Level 3
        self.spectral_sentinel = (
            SpectralSentinel(
                check_interval=spectral_interval,
                sensitivity=spectral_sensitivity,
                layer_stride=layer_stride,
            )
            if level >= 3 else None
        )

        # Cache checkpoint for rollback
        self._checkpoint = None

    def snapshot(self, cache, prompt_len):
        """Take initial fingerprint after encoding. Call once before generation.

        Args:
            cache: HuggingFace cache object (DynamicCache, etc.)
            prompt_len: Number of encoding tokens (positions to protect)
        """
        t0 = time.perf_counter()

        cache_layers = _get_cache_layers(cache)
        self._layers_monitored = sum(1 for l in cache_layers if l is not None)

        self.position_guard.snapshot(cache_layers, prompt_len)

        if self.norm_sentinel:
            self.norm_sentinel.snapshot(cache_layers, prompt_len)

        if self.spectral_sentinel:
            self.spectral_sentinel.snapshot(cache_layers, prompt_len)

        self._overhead_ms += (time.perf_counter() - t0) * 1000

    def checkpoint(self, cache):
        """Save a deep copy of the cache for potential rollback.

        WARNING: This doubles memory usage. Only use when rollback is needed.
        """
        import copy
        t0 = time.perf_counter()

        cache_layers = _get_cache_layers(cache)
        self._checkpoint = []
        for layer in cache_layers:
            if layer is not None:
                k, v = layer
                if torch is not None and hasattr(k, 'clone'):
                    self._checkpoint.append((k.clone(), v.clone()))
                else:
                    self._checkpoint.append((k.copy(), v.copy()))
            else:
                self._checkpoint.append(None)

        self._overhead_ms += (time.perf_counter() - t0) * 1000

    def rollback(self, cache):
        """Restore cache from checkpoint. Returns True if successful."""
        if self._checkpoint is None:
            return False

        cache_layers = _get_cache_layers(cache)

        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            for i, saved in enumerate(self._checkpoint):
                if saved is not None and i < len(cache.key_cache):
                    k_saved, v_saved = saved
                    cache.key_cache[i] = k_saved.clone() if hasattr(k_saved, 'clone') else k_saved.copy()
                    cache.value_cache[i] = v_saved.clone() if hasattr(v_saved, 'clone') else v_saved.copy()
            return True

        if hasattr(cache, 'layers'):
            for i, saved in enumerate(self._checkpoint):
                if saved is not None and i < len(cache.layers):
                    k_saved, v_saved = saved
                    cache.layers[i].keys = k_saved.clone() if hasattr(k_saved, 'clone') else k_saved.copy()
                    cache.layers[i].values = v_saved.clone() if hasattr(v_saved, 'clone') else v_saved.copy()
            return True

        return False

    def verify(self, cache):
        """Check cache integrity. Call before each generation step.

        Returns list of Violation objects (empty if clean).
        """
        t0 = time.perf_counter()
        step_violations = []

        cache_layers = _get_cache_layers(cache)

        # Level 1: Position hash
        step_violations.extend(self.position_guard.verify(cache_layers, self.step))

        # Level 2: Norm sentinel
        if self.norm_sentinel:
            step_violations.extend(self.norm_sentinel.verify(cache_layers, self.step))

        # Level 3: Spectral sentinel
        if self.spectral_sentinel:
            step_violations.extend(self.spectral_sentinel.verify(cache_layers, self.step))

        self.violations.extend(step_violations)
        self._overhead_ms += (time.perf_counter() - t0) * 1000

        return step_violations

    def update(self, cache):
        """Register a new generation token as legitimate. Call after each step."""
        t0 = time.perf_counter()
        self.step += 1

        cache_layers = _get_cache_layers(cache)

        if self.norm_sentinel:
            self.norm_sentinel.update(cache_layers)

        if self.spectral_sentinel:
            self.spectral_sentinel.update(cache_layers)

        self._overhead_ms += (time.perf_counter() - t0) * 1000

    def report(self):
        """Generate final integrity report."""
        return IntegrityReport(
            violations=self.violations,
            steps_monitored=self.step,
            layers_monitored=self._layers_monitored,
            level=self.level,
            overhead_ms=self._overhead_ms,
            clean=len(self.violations) == 0,
        )


# ===================================================================
# Convenience: wrap a generation loop
# ===================================================================

def monitored_generate(model, tokenizer, input_ids, max_tokens=200,
                       monitor_level=2, on_violation="warn"):
    """Generate tokens with cache integrity monitoring.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        input_ids: Encoded input (batch, seq_len)
        max_tokens: Maximum tokens to generate
        monitor_level: CIM detection level (1, 2, 3)
        on_violation: "warn" (print + continue), "stop" (halt generation),
                      "rollback" (restore checkpoint + halt)

    Returns:
        (text, report) tuple
    """
    if torch is None:
        raise ImportError("PyTorch required for monitored_generate")

    monitor = CacheIntegrityMonitor(level=monitor_level)

    # Encode
    with torch.no_grad():
        enc_out = model(input_ids, use_cache=True)
    cache = enc_out.past_key_values
    prompt_len = input_ids.shape[1]

    # Snapshot + optional checkpoint
    monitor.snapshot(cache, prompt_len)
    if on_violation == "rollback":
        monitor.checkpoint(cache)

    # First token
    next_token = enc_out.logits[:, -1:, :].argmax(dim=-1)
    del enc_out
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    tokens = [next_token.item()]
    eos = tokenizer.eos_token_id

    # Generation loop with monitoring
    for _ in range(max_tokens - 1):
        # Verify before generating
        violations = monitor.verify(cache)

        if violations:
            if on_violation == "stop":
                print(f"[CIM] HALTED: {violations[0].description}")
                break
            elif on_violation == "rollback":
                print(f"[CIM] ROLLBACK: {violations[0].description}")
                monitor.rollback(cache)
                break
            else:  # warn
                for v in violations:
                    print(f"[CIM] WARNING: {v.description}")

        # Generate one token
        with torch.no_grad():
            out = model(next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        tid = next_token.item()
        tokens.append(tid)

        # Register new token as legitimate
        monitor.update(cache)

        if tid == eos:
            break

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text, monitor.report()
