#!/usr/bin/env python3
"""Mutation harness. Every mutant must turn the suite red, or the test is decorative."""
import pathlib, shutil, subprocess, sys, tempfile
SRC = pathlib.Path(__file__).parent / "spectral_bands.py"
MUTANTS = [
    ("aperture never closes",        'bool(g >= APERTURE_MIN_GAMMA)', 'True'),
    ("aperture threshold moves",     'APERTURE_MIN_GAMMA = 2.0',      'APERTURE_MIN_GAMMA = 1.0'),
    ("closed band reads as usable",  'bool(not inverted and aperture_ok(m, n)["open"])', 'True'),
    ("conventions collapse",         'b = min(m, n) / max(m, n)',     'b = max(m, n) / min(m, n)'),
    ("gamma convention flips",       'g = max(m, n) / min(m, n)',     'g = min(m, n) / max(m, n)'),
    ("GD uses median of squares",    'sigma_median = float(np.median(sv))', 'sigma_median = float(np.median(sv**2))'),
    ("sigma_from is ignored",        'elif sigma_from == "lower":',   'elif False:'),
    ("bad sigma_from passes",        'raise ValueError(f"sigma_from must be',  'pass  # ('),
    ("stack ignores dim mismatch",   'raise ValueError(f"matrix {i} has',      'pass  # ('),
    ("null loses its shape",         '"m": m, "n": n, "sigma_from": sigma_from', '"m": 0, "n": 0, "sigma_from": sigma_from'),
    ("null trials not recorded",     '"trials": trials,',             '"trials": 0,'),
    ("verdict drops its null",       '"null": null,',                 '"null": None,'),
]
orig = SRC.read_text(); bak = tempfile.mktemp(suffix=".py"); shutil.copy(SRC, bak)
survivors = []
try:
    for name, old, new in MUTANTS:
        if old not in orig:
            print(f"  SKIP  {name}: anchor missing — harness is stale"); survivors.append(name); continue
        SRC.write_text(orig.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", "test_spectral_bands.py", "-q"],
                           cwd=SRC.parent, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  SURVIVED  {name}"); survivors.append(name)
        else:
            print(f"  killed    {name}")
finally:
    SRC.write_text(orig)
print()
if survivors:
    print(f"{len(survivors)} survived — those mutations are invisible to the suite"); sys.exit(1)
print(f"all {len(MUTANTS)} mutants killed")
