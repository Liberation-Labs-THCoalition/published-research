# Permutation Test Fix: Valence Probe (Audit Finding UM-B2)

## Bug Description

`mp_probe_recompute.py` performs a permutation test on valence probe R² by
shuffling valence labels at the **trial level**.  However, valence is constant
within each emotion group -- every "panicked" trial maps to valence = -0.90,
every "elated" trial maps to valence = +0.95, etc.  Trial-level shuffling
therefore breaks this grouping structure and creates artificially extreme null
distributions (each permutation scatters the 30 distinct valence values across
hundreds of trials randomly, destroying within-group correlation that exists
even under the null).

The correct null hypothesis is: "there is no mapping from emotion identity to
valence."  This is tested by permuting the 30 emotion-to-valence assignments
and propagating each shuffled assignment to all trials belonging to that
emotion.

## Fix

Replace the trial-level shuffle inside the permutation loop with
emotion-level permutation:

```python
# CORRECT: permute at emotion level, propagate to trials
unique_emotions = np.unique(emotions)
emotion_valence_map = {e: valences[emotions == e][0] for e in unique_emotions}

# In permutation loop:
shuffled_emotion_vals = np.random.permutation(list(emotion_valence_map.values()))
perm_map = dict(zip(unique_emotions, shuffled_emotion_vals))
v_perm = np.array([perm_map[e] for e in emotions])
```

`v_perm` then replaces the old `np.random.permutation(valences)` call (or
equivalent) as the shuffled target vector for fitting the probe under the null.

## Impact on Paper Claims

The valence probe result in the paper is a **null finding** -- valence
R² < 0, meaning the probe does not extract valence information above chance.
Because the original (biased) permutation test was being compared against a
null result, the inflated p-value does not change the qualitative conclusion.
The paper's claims are unaffected.

## Audit Reference

This fix addresses audit finding **UM-B2** (trial-level vs. emotion-level
permutation granularity).
