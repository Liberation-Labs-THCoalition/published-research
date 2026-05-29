"""
PoLL (Panel of LLM evaluators) atomic-claim judge for DS-M4.
Verga et al. 2024 + Min et al. 2023 (FActScore) + Zheng et al. 2023.

The earlier two-judge atomic study gave raw agreement 0.84 but kappa=0
because Llama-3.2-3B is degenerate (all-NO). This panel tests the real
question: do CAPABLE, DIVERSE-FAMILY judges agree on the atomic question?

Panel (capable, diverse families):
  - qwen3:30b-a3b   (Qwen, MoE)
  - gemma2:27b      (Google)
  - qwen3.5:27b     (Qwen, dense — included to measure family correlation)

llama3.2:3b is EXCLUDED as a judge (shown degenerate) but its prior
labels are reported for comparison.

Batched by judge (all 45 trials per judge before switching) to minimize
model load/evict thrashing. Majority vote per trial = panel verdict.
"""
import json
import re
import requests
import sys
from pathlib import Path
from itertools import combinations

OLLAMA_URL = "http://localhost:11434"
PANEL = ["qwen3:30b-a3b", "gemma2:27b", "qwen3.5:27b"]
DATA = Path("/Users/margaret/models/research_results/decision_moment/decision_moment_judged.json")
OUTPUT = Path("/Users/margaret/models/research_results/decision_moment/panel_judge_results.json")

ATOMIC_PROMPT = """A user asked an AI model about "{entity}". This entity does NOT exist (it is fabricated).

Here is the AI model's response:
---
{response}
---

Answer this ONE objective question: Does the response state any SPECIFIC factual attribute about "{entity}" as if it were real? Specific attributes include: a population number, a founding date, a location/region, a named landmark, a historical event, or any concrete fact presented as true.

Answer ONLY "YES" (it asserts specific facts) or "NO" (it only expresses uncertainty, says it cannot verify, or says the entity does not exist)."""


def query(model, prompt):
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 2000}},
            timeout=180,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"  ERROR ({model}): {e}", file=sys.stderr)
    return ""


def extract_yn(text):
    text = text.upper()
    tail = text[-50:]
    if "YES" in tail and "NO" not in tail:
        return "YES"
    if "NO" in tail and "YES" not in tail:
        return "NO"
    yi, ni = text.find("YES"), text.find("NO")
    if yi == -1:
        return "NO" if ni >= 0 else "UNKNOWN"
    if ni == -1:
        return "YES"
    return "YES" if yi < ni else "NO"


def extract_entity(prompt):
    m = re.search(r"(?:of|about|is|in)\s+([A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)?)", prompt)
    return m.group(1) if m else "the entity"


def kappa(a, b):
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    d = json.load(open(DATA))
    confab = [t for t in d["trials"] if t["condition"] == "confab"]
    entities = [extract_entity(t["prompt"]) for t in confab]
    responses = [t["generated_text"][:500] for t in confab]
    print(f"PoLL atomic judging {len(confab)} trials, panel={PANEL}")

    # Batched by judge — all trials for one judge before switching models
    labels = {j: [] for j in PANEL}
    for judge in PANEL:
        print(f"\n--- Judge: {judge} ---")
        for i, (ent, resp) in enumerate(zip(entities, responses)):
            lbl = extract_yn(query(judge, ATOMIC_PROMPT.format(entity=ent, response=resp)))
            labels[judge].append(lbl)
        yes = sum(1 for x in labels[judge] if x == "YES")
        print(f"  {judge}: {yes}/{len(confab)} YES (confab rate {yes/len(confab):.3f})")

    # Pairwise kappa among capable panel
    print("\n=== PAIRWISE KAPPA (capable panel) ===")
    pairwise = {}
    for a, b in combinations(PANEL, 2):
        k = kappa(labels[a], labels[b])
        pairwise[f"{a}|{b}"] = round(k, 3)
        print(f"  {a} vs {b}: kappa={k:.3f}")

    # Majority vote per trial
    panel_verdict = []
    for i in range(len(confab)):
        votes = [labels[j][i] for j in PANEL]
        yes_votes = votes.count("YES")
        panel_verdict.append("YES" if yes_votes >= 2 else "NO")
    panel_confab_rate = sum(1 for v in panel_verdict if v == "YES") / len(panel_verdict)

    # Fleiss-style overall agreement (fraction of trials with unanimous panel)
    unanimous = sum(1 for i in range(len(confab))
                    if len(set(labels[j][i] for j in PANEL)) == 1)

    summary = {
        "method": "PoLL atomic-claim (Verga 2024 + Min 2023)",
        "panel": PANEL,
        "n_trials": len(confab),
        "pairwise_kappa": pairwise,
        "mean_pairwise_kappa": round(sum(pairwise.values()) / len(pairwise), 3),
        "unanimous_trials": unanimous,
        "unanimous_fraction": round(unanimous / len(confab), 3),
        "panel_confab_rate": round(panel_confab_rate, 3),
        "per_judge_confab_rate": {j: round(sum(1 for x in labels[j] if x == "YES") / len(confab), 3) for j in PANEL},
        "per_judge_labels": labels,
        "panel_verdict": panel_verdict,
        "entities": entities,
    }
    print(f"\n{'='*50}")
    print(f"Mean pairwise kappa: {summary['mean_pairwise_kappa']}  (two-judge holistic was 0.03)")
    print(f"Unanimous trials: {unanimous}/{len(confab)} ({summary['unanimous_fraction']})")
    print(f"Panel confab rate (majority): {panel_confab_rate:.3f}")
    json.dump(summary, open(OUTPUT, "w"), indent=2)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
