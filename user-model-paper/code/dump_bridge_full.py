#!/usr/bin/env python3
"""Dump full bridge PCA, F-ratio, and PC1 scores for paper verification."""
import json, os

# Qwen bridge
qwen_path = os.path.expanduser("~/KV-Experiments/results/emotion_geometry_bridge/emotion_bridge_summary.json")
with open(qwen_path) as f:
    q = json.load(f)

print("=== QWEN TOP-LEVEL KEYS ===")
print(list(q.keys()))

# Show all non-trial keys
for key in q:
    if key == "trials":
        print("\n=== trials: %d items ===" % len(q[key]))
        continue
    if key == "bridge_analysis":
        print("\n=== bridge_analysis keys ===")
        ba = q[key]
        print(list(ba.keys()))
        # Show one layer's full structure
        if "35" in ba:
            print("\n=== BRIDGE L35 FULL ===")
            print(json.dumps(ba["35"], indent=2))
        if "3" in ba:
            print("\n=== BRIDGE L3 FULL ===")
            print(json.dumps(ba["3"], indent=2))
        if "55" in ba:
            print("\n=== BRIDGE L55 FULL ===")
            print(json.dumps(ba["55"], indent=2))
        if "63" in ba:
            print("\n=== BRIDGE L63 FULL ===")
            print(json.dumps(ba["63"], indent=2))
        if "summary" in ba:
            print("\n=== BRIDGE SUMMARY ===")
            print(json.dumps(ba["summary"], indent=2))
        continue
    print("\n=== %s ===" % key)
    val = q[key]
    if isinstance(val, (dict, list)):
        print(json.dumps(val, indent=2))
    else:
        print(val)
