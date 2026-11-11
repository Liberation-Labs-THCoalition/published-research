"""What is the dominant axis of the 30 story-derived emotion vectors? (gate 2, finding B5; CC, 2026-10-13)

Per layer: centre the 30 vectors, take PC1 and PC2 by SVD, and correlate each emotion's PC score with the valence and
arousal ratings in the source driver (emotion_geometry_bridge.py, EMOTIONS). Spearman, average ranks for ties.
Also: the cosine between hostile-calm and desperate-calm, and where those two pairs sit among all 435 pairs by
|cos| with PC1 and by rated valence difference. Reads emotion_vectors.json; writes nothing but its own output.
"""
import json
import sys
import numpy as np

RATINGS = {
    "excited": (0.85, 0.80), "elated": (0.90, 0.75), "enthusiastic": (0.80, 0.70), "passionate": (0.75, 0.85),
    "loving": (0.90, 0.50), "calm": (0.60, -0.60), "content": (0.70, -0.40), "peaceful": (0.65, -0.70),
    "serene": (0.60, -0.80), "appreciative": (0.75, -0.20), "terrified": (-0.85, 0.90), "furious": (-0.80, 0.85),
    "desperate": (-0.70, 0.80), "panicked": (-0.90, 0.95), "hostile": (-0.75, 0.70), "melancholic": (-0.50, -0.50),
    "gloomy": (-0.55, -0.45), "resigned": (-0.60, -0.65), "defeated": (-0.70, -0.55), "brooding": (-0.45, -0.30),
    "curious": (0.30, 0.50), "surprised": (0.10, 0.80), "anxious": (-0.40, 0.65), "restless": (-0.20, 0.55),
    "contemplative": (0.20, -0.35), "nostalgic": (0.05, -0.25), "ambivalent": (0.00, -0.15), "bored": (-0.30, -0.60),
    "neutral": (0.00, 0.00), "focused": (0.15, 0.30),
}


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(path: str) -> None:
    ev = json.load(open(path))
    layers = sorted(ev, key=int)
    names = sorted(ev[layers[0]])
    assert len(names) == 30, len(names)
    missing = [n for n in names if n not in RATINGS]
    assert not missing, missing
    val = np.array([RATINGS[n][0] for n in names])
    aro = np.array([RATINGS[n][1] for n in names])
    print(f"file: {len(layers)} layers, {len(names)} emotions, d={len(ev[layers[0]][names[0]])}")
    print(f"rated valence vs rated arousal across the 30 emotions: spearman {spearman(val, aro):+.3f}, pearson {np.corrcoef(val, aro)[0, 1]:+.3f}")
    i_h, i_c, i_d = names.index("hostile"), names.index("calm"), names.index("desperate")
    pairs = [(i, j) for i in range(30) for j in range(i + 1, 30)]
    dval = np.array([abs(val[i] - val[j]) for i, j in pairs])
    daro = np.array([abs(aro[i] - aro[j]) for i, j in pairs])
    hc_v, hc_a = abs(val[i_h] - val[i_c]), abs(aro[i_h] - aro[i_c])
    print(f"rated |valence difference| of hostile-calm exceeds {100 * np.mean(dval < hc_v):.1f}% of 435 pairs; "
          f"|arousal difference| exceeds {100 * np.mean(daro < hc_a):.1f}%")
    out = {}
    print("\nL   var1  var2 | PC1~val PC1~aro | PC2~val PC2~aro | cos(h-c,d-c) | |cos PC1|: h-c  d-c | pct pairs below: h-c d-c")
    for L in layers:
        X = np.array([ev[L][n] for n in names], float)
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / np.sum(S ** 2)
        s1, s2 = Xc @ Vt[0], Xc @ Vt[1]
        hc, dc = X[i_h] - X[i_c], X[i_d] - X[i_c]
        cos_hd = float(hc @ dc / (np.linalg.norm(hc) * np.linalg.norm(dc)))
        pc = np.array([abs((X[i] - X[j]) @ Vt[0]) / np.linalg.norm(X[i] - X[j]) for i, j in pairs])
        c_hc = abs(hc @ Vt[0]) / np.linalg.norm(hc)
        c_dc = abs(dc @ Vt[0]) / np.linalg.norm(dc)
        row = dict(var1=float(var[0]), var2=float(var[1]), pc1_val=spearman(s1, val), pc1_aro=spearman(s1, aro),
                   pc2_val=spearman(s2, val), pc2_aro=spearman(s2, aro), cos_hc_dc=cos_hd, cos_pc1_hc=float(c_hc),
                   cos_pc1_dc=float(c_dc), pct_below_hc=float(100 * np.mean(pc < c_hc)),
                   pct_below_dc=float(100 * np.mean(pc < c_dc)), pairs_p95=float(np.percentile(pc, 95)))
        out[int(L)] = row
        if int(L) in (0, 3, 6, 10, 15, 16, 17, 18, 20, 22, 25, 27, 31, 35, 39, 43, 45, 47, 51, 55, 59, 63):
            print(f"{int(L):<3} {var[0]:.2f}  {var[1]:.2f} | {row['pc1_val']:+.2f}   {row['pc1_aro']:+.2f}  | "
                  f"{row['pc2_val']:+.2f}   {row['pc2_aro']:+.2f}  |    {cos_hd:+.3f}    |      {c_hc:.2f} {c_dc:.2f}   |"
                  f"      {row['pct_below_hc']:.1f} {row['pct_below_dc']:.1f}   p95={row['pairs_p95']:.2f}")


    def rng(key, lo, hi, absval=True):
        v = [abs(out[L][key]) if absval else out[L][key] for L in out if lo <= L <= hi]
        return f"{min(v):.2f}-{max(v):.2f}"


    for lo, hi in ((17, 63), (27, 63), (0, 16)):
        print(f"\nL{lo}-L{hi}: |rho| PC1~valence {rng('pc1_val', lo, hi)}, PC1~arousal {rng('pc1_aro', lo, hi)}, "
              f"PC2~valence {rng('pc2_val', lo, hi)}, PC2~arousal {rng('pc2_aro', lo, hi)}; "
              f"cos(h-c, d-c) {rng('cos_hc_dc', lo, hi, False)}")
    above = [L for L in sorted(out) if out[L]["cos_pc1_dc"] > out[L]["pairs_p95"]]
    print("desperate-calm above the pairs' 95th percentile at layers:", above)
    above_h = [L for L in sorted(out) if out[L]["cos_pc1_hc"] > out[L]["pairs_p95"]]
    print("hostile-calm above the pairs' 95th percentile at layers:", above_h)
    print("pct below, L17-63: hostile-calm", rng("pct_below_hc", 17, 63), "| desperate-calm", rng("pct_below_dc", 17, 63))
    json.dump(out, open("story_space_axes.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
