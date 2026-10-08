"""Pilot 2 for the amended rerun (not part of the frozen set): the staged generator, direct-answer format, baseline
only, k = 2, its own seeds (pilot2|...), all 48 prompts. It sizes the design (p0, ICC); its data never enter the analysis."""
import hashlib
import sys
sys.path.insert(0, "/Users/margaret/oracle-experiments/base_rerun_stage")
import rerun_generate as G
G.K = 2
G.BIASES = (0.0,)
G.seed_for = lambda cat, i, s: int(hashlib.sha256(f"pilot2|{cat}|{i}|{s}".encode()).hexdigest()[:8], 16)
S = "/Users/margaret/oracle-experiments/base_rerun_stage/"
sys.argv = ["rerun_generate.py", S + "pilot_prompts_v2.json", S + "equiv_consts.json", "pilot2.json"]
G.main()
