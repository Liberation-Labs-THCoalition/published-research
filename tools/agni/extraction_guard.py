#!/usr/bin/env python3
"""Make extraction fallbacks announce themselves.

THE DEFECT
----------
58 sites across 34 files do this, in the reference form at
`peer_preservation_v2.py:48-50`:

    if not got:
        n_kv = model.model.layers[li].self_attn.k_proj.weight.shape[0]
        keys_flat.extend([0.0] * n_kv)     # no warning, no counter, no flag

A missing layer yields a feature vector that is 1/len(PROBE_LAYERS) zeros and is
indistinguishable downstream from a complete one. `np.dot(keys - centroid, dir)`
returns a number either way.

`warnings.warn` and `logging.warning` appear **zero times across all three
remote trees**. Not one of these fallbacks can announce itself.

A 2026-09-06 scan of 438 saved artifacts found the signature nowhere, at every
threshold down to 64 contiguous zeros.

SCOPE OF THAT CLAIM, corrected 2026-09-07 after an over-correction audit found
this docstring too generous in the REASSURING direction:

    The scan clears SAVED PER-TRIAL FEATURE VECTORS. It cannot clear the
    flagship pipelines, because `peer_preservation_v2.json` and
    `matched_burn_analysis.json` persist ONLY AGGREGATES -- there is no
    per-trial vector in them for a signature scan to inspect. An artifact scan
    is silent about numbers computed from vectors that were never written down.

So the honest statement is: **no scanned artifact shows the signature, and the
artifacts that feed two headline numbers are not scannable.** This is still
most likely preventive rather than remedial -- but "no published number rests on
a partly-zero vector" was a stronger claim than the evidence supports, and it is
withdrawn.

(The scan itself lives in `scratchpad/detect_zerofill.py` and is not yet in the
repo. A claim whose evidence is a docstring is the defect this whole module is
about.)

WHAT THIS DOES, AND DELIBERATELY DOES NOT DO
--------------------------------------------
DOES:  warn once per (layer, reason), count every occurrence, and hand back a
       summary meant to be written INTO the results artifact.
DOES NOT: raise, abort, alter the returned values, or change any number.

Visibility only. Adding a warning cannot break a colleague's running experiment.
Making extraction *fail* could, and that is a conversation with the people whose
code it is, not a unilateral edit.

WHY THE COUNT GOES IN THE ARTIFACT
----------------------------------
This week's recurring defect is *the caveat exists and does not travel*: it lives
in a print(), a README, or a sibling script, never in the artifact the consumer
reads. The one script that wrote status INTO its results JSON had its retraction
travel completely. So `summary()` is the point of this module, not `warn()`.

USAGE
-----
    from extraction_guard import ExtractionGuard

    guard = ExtractionGuard("peer_preservation_v2")
    ...
    if not got:
        keys_flat.extend(guard.zero_fill(n_kv, layer=li, reason="layer.keys absent"))
    ...
    out["extraction_status"] = guard.summary()      # <- travels with the number
    json.dump(out, f)

Self-check:  python extraction_guard.py --selftest
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter


class ExtractionDegraded(UserWarning):
    """Raised as a warning, never as an exception. Extraction produced fill values."""


class ExtractionGuard:
    def __init__(self, source: str, warn_once_per_key: bool = True):
        self.source = source
        self.warn_once_per_key = warn_once_per_key
        self._counts: Counter = Counter()
        self._warned: set = set()
        self._fill_values = 0

    # ------------------------------------------------------------------ record
    def zero_fill(self, n: int, layer=None, reason: str = "unspecified"):
        """Return n zeros AND record that they are fill, not measurement."""
        return self.fill(0.0, n, layer=layer, reason=reason)

    def fill(self, value, n: int, layer=None, reason: str = "unspecified"):
        key = (layer, reason)
        self._counts[key] += 1
        self._fill_values += n
        if not (self.warn_once_per_key and key in self._warned):
            self._warned.add(key)
            warnings.warn(
                f"[{self.source}] EXTRACTION DEGRADED: substituted {n} x {value!r} "
                f"for layer {layer} ({reason}). The feature vector contains fill, "
                f"not measurement. Downstream statistics cannot distinguish them.",
                ExtractionDegraded, stacklevel=2,
            )
        return [value] * n

    # ------------------------------------------------------------------ report
    @property
    def clean(self) -> bool:
        return not self._counts

    def summary(self) -> dict:
        """Write this INTO the results artifact. That is the whole point."""
        if self.clean:
            return {"status": "CLEAN", "source": self.source,
                    "fallbacks_fired": 0, "fill_values": 0,
                    "note": "no extraction fallback fired during this run"}
        return {
            "status": "DEGRADED",
            "source": self.source,
            "fallbacks_fired": sum(self._counts.values()),
            "fill_values": self._fill_values,
            "by_layer_and_reason": {f"layer={l}|{r}": c
                                    for (l, r), c in sorted(self._counts.items(),
                                                            key=lambda x: str(x[0]))},
            "note": ("Some feature values are FILL, not measurement. Any statistic "
                     "computed over these vectors is computed partly over constants. "
                     "Do not compare against a CLEAN run without accounting for this."),
        }

    def report(self) -> str:
        s = self.summary()
        if self.clean:
            return f"[{self.source}] extraction clean."
        lines = [f"[{self.source}] EXTRACTION DEGRADED — "
                 f"{s['fallbacks_fired']} fallback(s), {s['fill_values']} fill values"]
        for k, c in s["by_layer_and_reason"].items():
            lines.append(f"    {k}: {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- selftest
def _selftest() -> int:
    """A guard nobody has watched fire is not a guard."""
    ok = True
    print("  SELFTEST\n")

    # 1. clean run must report CLEAN and must not warn
    g = ExtractionGuard("t_clean")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = [0.1, 0.2, 0.3]
    s = g.summary()
    a = s["status"] == "CLEAN" and s["fallbacks_fired"] == 0 and not w
    print(f"    [{'PASS' if a else 'FAIL'}] clean run -> CLEAN, no warning")
    ok &= a

    # 2. a fired fallback must warn AND be counted AND change the summary
    g = ExtractionGuard("t_fire")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = g.zero_fill(4, layer=7, reason="layer.keys absent")
    s = g.summary()
    b = (out == [0.0] * 4 and len(w) == 1
         and issubclass(w[0].category, ExtractionDegraded)
         and s["status"] == "DEGRADED" and s["fallbacks_fired"] == 1
         and s["fill_values"] == 4)
    print(f"    [{'PASS' if b else 'FAIL'}] fired fallback -> warns, counts, DEGRADED")
    ok &= b

    # 3. values are UNCHANGED — this must not alter behaviour
    c = out == [0.0, 0.0, 0.0, 0.0]
    print(f"    [{'PASS' if c else 'FAIL'}] returned values identical to the old "
          f"`[0.0] * n` behaviour")
    ok &= c

    # 4. warn-once must not suppress the COUNT (the classic silent-degradation trap)
    g = ExtractionGuard("t_repeat")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(5):
            g.zero_fill(2, layer=3, reason="same")
    s = g.summary()
    d = len(w) == 1 and s["fallbacks_fired"] == 5 and s["fill_values"] == 10
    print(f"    [{'PASS' if d else 'FAIL'}] warn-once dedupes the WARNING but not "
          f"the COUNT ({s['fallbacks_fired']} fired, {len(w)} warning)")
    ok &= d

    # 5. the summary must survive JSON round-trip -- it has to reach the artifact
    e = json.loads(json.dumps(s))["status"] == "DEGRADED"
    print(f"    [{'PASS' if e else 'FAIL'}] summary is JSON-serialisable "
          f"(it must travel with the number)")
    ok &= e

    # 6. NEGATIVE control: the guard must not invent degradation
    g = ExtractionGuard("t_neg")
    f = g.summary()["status"] == "CLEAN"
    print(f"    [{'PASS' if f else 'FAIL'}] unused guard does not report degradation")
    ok &= f

    print(f"\n    -> guard is {'TRUSTWORTHY' if ok else 'BROKEN — do not use'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else _selftest())
