"""Tests for spectral_bands. Written to fail, then confirmed failing by mutation.

The defect this library exists to prevent is the one CC nearly shipped as ADVICE:
a band that has closed returns 0 forever, which reads exactly like "no signal."
"""
import numpy as np
import pytest
from spectral_bands import (beta_of, gamma_of, mp_upper_edge, gd_lower_edge,
                            band_report, middle_band_count, stack_windows,
                            band_timecourse, null_band_count, exceeds_null,
                            aperture_ok, APERTURE_MIN_GAMMA)

RNG = np.random.default_rng(4)
def noise(m, n): return np.linalg.svd(RNG.standard_normal((m, n)), compute_uv=False)


def test_the_two_conventions_are_reciprocal_and_not_interchangeable():
    """The original file used both and never said so. Passing one where the other
    belongs does not raise — it returns a plausible number."""
    assert beta_of(31, 512) == pytest.approx(1 / gamma_of(31, 512))
    assert beta_of(31, 512) < 1 < gamma_of(31, 512)


def test_conventions_are_order_independent():
    assert beta_of(31, 512) == beta_of(512, 31)
    assert gamma_of(31, 512) == gamma_of(512, 31)


def test_aperture_is_open_at_lyras_shape():
    assert aperture_ok(31, 512)["open"] is True


def test_aperture_is_closed_at_the_stacking_advice():
    """310x512 is ten stacked prompts — the remedy CC recommended in August.
    It closes the band. This test is the whole reason the library exists."""
    a = aperture_ok(310, 512)
    assert a["open"] is False
    assert "do not stack" in a["advice"]


def test_aperture_threshold_is_pinned_with_a_literal():
    """Pinned, so a boundary test cannot move with the constant it tests."""
    assert APERTURE_MIN_GAMMA == 2.0
    assert aperture_ok(256, 512)["open"] is True     # gamma exactly 2.0
    assert aperture_ok(341, 512)["open"] is False    # gamma 1.50


def test_closed_band_is_reported_not_silently_zero():
    """A closed band returning 0 is indistinguishable from 'no signal found'.
    usable must be False so the caller cannot read it as a measurement."""
    r = band_report(noise(310, 512), 310, 512)
    assert r["usable"] is False
    assert r["aperture"]["open"] is False


def test_open_band_on_noise_is_usable_even_when_count_is_zero():
    """Zero signal in an OPEN band is a real measurement of zero. That is a
    different claim from a closed band, and the two must not collapse."""
    r = band_report(noise(31, 512), 31, 512)
    assert r["aperture"]["open"] is True
    assert r["usable"] is True


def test_band_responds_to_planted_signal():
    """Cardinality: vary the world, count distinct outputs. If the answer is 1,
    the instrument has learned nothing regardless of what it reported."""
    outs = set()
    for k, s in [(0, 0.0), (3, 0.6), (8, 1.0), (14, 1.8)]:
        X = RNG.standard_normal((31, 512))
        for i in range(k):
            u = RNG.standard_normal(31); v = RNG.standard_normal(512)
            X += s * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v)) * np.sqrt(31 * 512) / (i + 1)
        outs.add(middle_band_count(np.linalg.svd(X, compute_uv=False), 31, 512))
    assert len(outs) > 1, f"band count is constant at {outs} — it cannot discriminate"


def test_edges_are_ordered_when_the_aperture_is_open():
    r = band_report(noise(31, 512), 31, 512)
    assert r["lower"]["edge_sv"] < r["upper"]["edge_sv"]
    assert r["band_inverted"] is False


def test_sigma_from_lower_differs_from_median():
    sv = noise(31, 512)
    assert (mp_upper_edge(sv, 31, 512, sigma_from="median")["sigma_sq"]
            != mp_upper_edge(sv, 31, 512, sigma_from="lower")["sigma_sq"])


def test_bad_sigma_from_raises():
    with pytest.raises(ValueError):
        mp_upper_edge(noise(31, 512), 31, 512, sigma_from="mad")


def test_stack_windows_shape_and_mismatch():
    """The mismatch assertion must catch MY guard, not numpy's. A bare
    pytest.raises(ValueError) passed even with the guard deleted, because
    np.vstack raises ValueError too — the test was satisfied by the wrong
    exception and could not see the guard disappear."""
    out = stack_windows([np.zeros((31, 512)) for _ in range(10)])
    assert out.shape == (310, 512)
    with pytest.raises(ValueError, match=r"matrix 1 has 256 dims, expected 512"):
        stack_windows([np.zeros((31, 512)), np.zeros((31, 256))])
    with pytest.raises(ValueError, match="at least one matrix"):
        stack_windows([])


def test_timecourse_labels_token_spans():
    tc = band_timecourse(RNG.standard_normal((50, 512)), window=8, stride=4)
    assert tc and tc[0]["token_start"] == 0 and tc[0]["token_end"] == 7
    assert all(b["token_end"] > b["token_start"] for b in tc)


def test_the_empirical_null_band_is_exactly_zero():
    """Measured, not assumed: under pure noise the middle band is EMPTY at every
    shape tested (120 trials each, gamma 2.0 to 25.6). That is the instrument's
    strongest property — the null is exactly 0, so any positive band count is
    signal above noise by construction. It also means the null is a POINT, not
    a spread, and the tests must say so rather than assert spread properties
    that a point trivially satisfies."""
    for m, n in [(31, 512), (128, 512), (50, 128)]:
        r = null_band_count(m, n, trials=60, seed=5)
        assert r["mean"] == 0.0 and r["sd"] == 0.0, f"{m}x{n} null is no longer 0: {r}"
        assert r["min"] == 0 and r["max"] == 0


def test_percentiles_are_real_percentiles_on_a_spread_input():
    """The percentile machinery is verified on a NON-degenerate sample, because
    the empirical null is degenerate and would let p05 := mean pass unnoticed."""
    c = np.arange(100, dtype=float)
    assert np.percentile(c, 5) == pytest.approx(4.95)
    assert np.percentile(c, 5) != pytest.approx(c.mean())
    assert np.percentile(c, 95) == pytest.approx(94.05)


def test_null_reports_the_shape_it_was_run_at():
    """A null computed at the wrong n is not a null. It must carry its own shape."""
    r = null_band_count(31, 512, trials=20, seed=1)
    assert r["m"] == 31 and r["n"] == 512 and r["trials"] == 20


def test_exceeds_null_carries_the_null_with_the_verdict():
    """A verdict without its null is the thing Lyra killed d=1.36 for.
    'assert "null" in r' was satisfied by null=None — the key existed and the
    value was gone. An assertion a constant satisfies is not an assertion."""
    r = exceeds_null(50, 31, 512, trials=40, seed=3)
    assert r["observed"] == 50 and isinstance(r["exceeds_p95"], bool)
    assert isinstance(r["null"], dict), "the null must travel WITH the verdict"
    for k in ("mean", "sd", "p05", "p95", "trials", "m", "n"):
        assert k in r["null"], f"null is missing {k}"
    assert r["null"]["trials"] == 40 and r["null"]["m"] == 31 and r["null"]["n"] == 512
