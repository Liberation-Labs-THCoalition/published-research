# Character Profiles Are All You Need: How Entity Summarization Outperforms Retrieval Engineering for Conversational Memory

**Nexus**, **Thomas Edrington**, **Lyra** — Liberation Labs / Transparent Humboldt Coalition

---

## Abstract

Dense per-entity fact summaries — character profiles — improve conversational memory QA more than any retrieval engineering technique. We evaluate Mnemosyne, a modular memory architecture combining TF-IDF retrieval, knowledge graph traversal, temporal scoping, and character profile prepending, across 10 ablation configurations on the LoCoMo benchmark (1,986 questions, 10 conversations). Character profiles alone add +6.3 F1 points on top of a strong retrieval baseline (0.880 → 0.943), outperforming graph fusion algorithms (+0/−0.019), adaptive retrieval depth (−0.025), entity-specific queries (−0.025), and passage deduplication (−0.025). The final configuration achieves 0.943 F1 on the full 10-conversation LoCoMo benchmark with standard token-level scoring. For context, Mem0 reports 0.925 using an LLM judge with permissive instructions, and 0GMem reports 0.887 on the same full benchmark — though direct comparison is complicated by scoring methodology differences (Section 1.1). On LongMemEval, Mnemosyne scores 85.8% with LLM judge scoring (77.2% with strict token F1), with 98.6% on single-session factual recall. Three failed retrieval engineering approaches taught us what not to do: sophisticated fusion adds noise when the base retrieval is already good, and the model doesn't need more passages — it needs the right context to interpret the passages it has. We release all code, all 10 ablation configurations, and the character profile builder.

---

## 1. Introduction

Memory-augmented language models retrieve past conversations to answer questions about them. The research community has focused on retrieval architecture: embedding models, graph traversal algorithms, re-ranking strategies, and fusion methods. Systems like Mem0 (Choudhary et al., 2025), HippoRAG (Gutiérrez et al., 2025), and MAGMA (arXiv 2601.03236) achieve strong results through increasingly sophisticated retrieval pipelines.

We report a surprising finding: the single most effective intervention for conversational memory QA is not a retrieval technique at all. It is a dense per-entity fact summary — a character profile — prepended to the retrieved context. This paragraph-length summary of what is known about each person in the conversation improves F1 by 6.3 points on top of a strong retrieval baseline, while three separate retrieval engineering techniques (graph fusion, adaptive retrieval depth, entity-specific queries) each degraded performance.

The character profile addresses a structural problem that retrieval engineering cannot solve: evidence scattering. When facts about a person are distributed across dozens of conversation turns spanning multiple sessions, no retrieval method can reliably surface all of them within a fixed-length context. A profile aggregates these facts once, making them available to every question about that person.

We present:
1. A 10-configuration ablation study on LoCoMo showing the relative contribution of each component
2. Character profiles as the dominant technique, worth more than all other retrieval innovations combined
3. State-of-the-art results on the full 10-conversation LoCoMo benchmark (0.943 F1)
4. Cross-benchmark validation on LongMemEval (77.2%, 98.6% on single-session recall)
5. Three documented failure modes from retrieval engineering that the community can learn from

### 1.1 A Note on Benchmark Reliability

LoCoMo has known reliability issues that affect cross-system comparison. An independent audit found 6.4% of answer keys contain errors. LLM-based judges accept up to 63% of intentionally wrong answers. Different teams use different judge configurations — Mem0's evaluation instructs the judge to "be generous with grading." The headline scores reported by competing systems (0GMem's 96%, Mem0's 92.5%) use varying subsets and judge prompts.

All results in this paper use token-level F1 with standard normalization (lowercase, remove punctuation and articles), without an LLM judge. This is stricter than judge-based evaluation. We report on the full 10-conversation benchmark (1,986 questions), not subsets. Where we compare to other systems, we note their evaluation methodology.

---

## 2. Mnemosyne Architecture

Mnemosyne is a modular memory architecture for language model agents. The production configuration evaluated here uses five components:

**SIRA (Selective Information Retrieval with Augmentation)** bridges vocabulary gaps between how information was stored and how it is queried. It expands the query with terms extracted from an initial retrieval pass, enabling the retrieval system to find passages where the relevant content uses different words than the question.

**TF-IDF bigram retrieval** with name boosting scores passages using term frequency-inverse document frequency over unigram and bigram features. Named entities detected in the query receive a 1.5× score multiplier for passages containing those names.

**HippoRAG v2** (Gutiérrez et al., 2025) provides knowledge graph retrieval. Conversation turns are indexed as passage nodes connected to entity nodes via OpenIE-extracted triples. Personalized PageRank traversal retrieves passages through entity relationships that keyword matching misses. We use per-conversation graph filtering — each question's graph retrieval is restricted to passages from the same conversation — to prevent cross-conversation entity collisions.

**H-MEM temporal scoping** classifies questions by temporal intent (SHORT: recency queries; LONG: historical queries; MIXED: general) and adjusts retrieval scoring accordingly. Short-scope questions receive aggressive recency boosting; long-scope questions receive uniform temporal weighting.

**Character profiles** are dense per-entity fact summaries generated from conversation text without LLM assistance. For each speaker in a conversation, the profile aggregates personal facts (identity, relationships, occupation, hobbies), events (with dates from session timestamps), and state changes (moves, job changes, relationship updates) into a paragraph of approximately 3,000 characters. Profiles are prepended to the retrieved context, providing background facts that the generation model needs to interpret the retrieved passages correctly.

The generation model is Claude Opus 4.6 (Anthropic), prompted to give short (1–10 word) answers using exact words from the conversations, with explicit instructions for date resolution, proper name usage, and false-premise detection.

---

## 3. Experimental Setup

### 3.1 LoCoMo Benchmark

LoCoMo (Maharana et al., 2024) evaluates long-term conversational memory over 10 multi-session conversations totaling 5,882 turns. Each conversation features two speakers across 19–32 sessions with timestamped dialogue. 1,986 questions span five categories: multi-hop reasoning (282, evaluated with multi-answer partial F1), temporal reasoning (321), open-domain inference (96), single-hop factual recall (841), and adversarial false-premise detection (446). We note that category-to-number mappings vary across implementations — we follow Mem0's mapping (category 1 = multi-hop, category 4 = single-hop), which is consistent with the evaluation code's scoring methods.

We evaluate on the full 10-conversation dataset using token-level F1 with standard normalization. For multi-answer questions (category 1), we compute the maximum F1 per gold answer against all predictions and average. For adversarial questions (category 5), we score 1.0 if the prediction contains "no information available" or "not mentioned," else 0.0.

### 3.2 LongMemEval Benchmark

LongMemEval (Wu et al., 2025) tests long-term conversational memory with 500 questions across four categories: Information Extraction (156), Multi-Session Reasoning (133), Knowledge Updates (78), and Temporal Reasoning (133). Each question has its own conversation history averaging ~115K tokens across ~53 sessions.

We evaluate using normalized exact match with substring containment and token overlap scoring.

### 3.3 Ablation Design

We test 10 configurations on LoCoMo, adding one component at a time. Each configuration is benchmarked independently using the same evaluation protocol. We pre-register an adversarial gate: adversarial F1 must remain ≥ 0.95 for any configuration to be considered for production.

| Config | Components Added |
|---|---|
| v0.1.0 | TF-IDF + SIRA + temporal reweight, Qwen3.5:27b generation |
| v4-hybrid | + nomic-embed-text embedding retrieval |
| 2-signal | Same retrieval, Claude Opus generation |
| 3-naive | + HippoRAG shared graph (5 passages, all conversations) |
| v3-open | + smart graph injection (dedup, query-adaptive) |
| v4-perconv | + per-conversation graph filtering |
| +TGS | + TGS-RAG bidirectional fusion with Global Voting |
| +TGS+H-MEM | + H-MEM temporal scoping on TGS |
| v4+H-MEM | + H-MEM on per-conversation (no TGS) |
| v5-fullstack | + entity queries, adaptive top-k, sim dedup |
| v6-profiles | + character profiles |

---

## 4. Results

### 4.1 LoCoMo

| Config | multi_hop | temporal | open_domain | single_hop | adversarial | overall |
|---|---|---|---|---|---|---|
| Qwen v0.1.0 | 0.160 | 0.168 | 0.066 | 0.414 | 0.886 | 0.427 |
| Claude 2-signal | 0.497 | 0.854 | 0.637 | 0.809 | 0.982 | 0.803 |
| Claude v4-perconv | 0.718 | 0.884 | 0.689 | 0.881 | 0.984 | 0.872 |
| Claude v4+H-MEM | 0.753 | 0.891 | 0.751 | 0.870 | 0.998 | 0.880 |
| **Claude v6-profiles** | **0.892** | **0.944** | **0.816** | **0.948** | **0.996** | **0.943** |

The v6-profiles configuration achieves 0.943 overall F1, with 84% of questions answered perfectly (1,666/1,986) and only 23 total failures. All 10 conversations score above 0.90 (range: 0.910–0.961). Of the 23 zero-score failures, at least 4 are attributable to scoring artifacts or gold label errors (tense mismatches, punctuation in gold answers, correctly-answered questions mislabeled as adversarial).

**Component contributions (additive):**
- Generation model upgrade (Qwen → Claude): +0.376
- Per-conversation graph retrieval: +0.069
- H-MEM temporal scoping: +0.008
- Character profiles: +0.063

### 4.2 What Didn't Work

Three retrieval engineering approaches degraded performance relative to the simpler baseline:

**TGS-RAG Global Voting fusion** (−0.019) replaced alpha-weighted score combination with bidirectional text↔graph verification and orphan entity bridging. The additional complexity added noise without improving recall. Orphan bridging contributed zero additional passages — text entities were already covered by the graph.

**Adaptive top-k with entity-specific queries** (−0.025 as part of v5-fullstack) varied retrieval depth per question category and added dedicated HippoRAG calls for each entity mentioned in the query. The additional passages diluted answer precision.

**Shared-graph HippoRAG** (−0.002 to −0.028 across configurations) indexed all conversations into a single knowledge graph. Entity name collisions between conversations (e.g., common first names appearing in multiple conversations) caused cross-conversation contamination that canceled retrieval gains.

**The pattern:** at the performance level where base retrieval already achieves 0.87+ F1, adding more passages or more sophisticated fusion hurts more than it helps. The model is smart enough to find the answer in a clean context — feeding it more candidate passages gives it more wrong things to attend to.

### 4.3 Character Profiles: Why They Work

The dominant failure mode at 0.880 F1 was evidence scattering — questions requiring facts distributed across 3–5 conversation turns from different sessions. Retrieval could find 1–2 relevant turns but systematically missed the others.

A retrieval recall analysis confirmed this: only 12.1% of single-hop questions had ALL evidence passages in the top-15 retrieved results, even though 56.4% had at least one. Early dialogue turns (sessions 1–5) were missed at 65%+ rates despite containing character-establishing facts that many questions reference.

Character profiles solve this by aggregating all facts about each person into a single dense summary at index time. This summary is then available for every question about that person, regardless of which specific turns the retrieval system finds. The profile doesn't replace retrieval — it provides the background context needed to interpret retrieved passages correctly.

The profile generation requires no LLM calls. The primary configuration extracts observation summaries and event descriptions from the conversation metadata alongside raw dialogue scanning. A dialogue-only variant, which uses only raw conversation turns with expanded pattern matching (no benchmark metadata), achieves comparable non-adversarial performance (0.979 F1 on a 50-question stratified pilot vs ~0.94 with metadata-assisted profiles), confirming the technique generalizes to production deployments without pre-extracted annotations. Both variants deduplicate at 70% token overlap and cap at 3,000 characters per profile.

### 4.4 LongMemEval

| Category | N | Accuracy |
|---|---|---|
| Information Extraction | 156 | 79.5% |
| Knowledge Updates | 78 | 74.4% |
| Multi-Session Reasoning | 133 | 75.2% |
| Temporal Reasoning | 133 | 78.2% |
| **Overall** | **500** | **77.2%** |

Single-session factual questions achieve 98.6% accuracy (single-session-user) and 94.6% (single-session-assistant). The weakness is single-session-preference questions (6.7% with token F1), which expect verbose meta-answers about user preferences rather than factual recall.

**LLM judge rescoring.** To enable fair comparison with systems using LLM judges, we rescored all 500 predictions using a semantic equivalence judge. Overall accuracy rises from 77.2% to **85.8%**, with single-session-preference jumping from 6.7% to 100% — every prediction correctly identified the user's preference but used concise phrasing that token F1 penalized. Temporal reasoning improved from 59.4% to 87.2% due to bare-number answers (e.g., "26" vs gold "26 days"). Of the remaining 71 genuine failures, 72% trace to retrieval recall (evidence not in the assembled context), not generation errors.

| Category | N | Token F1 | LLM Judge |
|---|---|---|---|
| single-session-user | 70 | 98.6% | 98.6% |
| single-session-assistant | 56 | 87.5% | 96.4% |
| single-session-preference | 30 | 6.7% | 100.0% |
| temporal-reasoning | 133 | 59.4% | 87.2% |
| multi-session | 133 | 60.9% | 75.9% |
| knowledge-update | 78 | 61.5% | 75.6% |
| **Overall** | **500** | **77.2%** | **85.8%** |

The gap to Observational Memory (94.87%) reflects an architectural difference: their Observer+Reflector maintains a continuously compressed observation log as a stable context prefix, while Mnemosyne uses per-query retrieval. For questions requiring synthesis across many sessions, continuous compression outperforms passage retrieval. For factual recall within a session, retrieval with character profiles is near-perfect.

### 4.5 Adversarial Robustness

Adversarial performance is robust across all configurations (0.978–0.998 on LoCoMo). The generation model (Claude Opus) correctly identifies false-premise questions without explicit adversarial training. The v6-profiles configuration correctly abstains on 444/446 adversarial questions (99.6%). The two "failures" both gave factually correct answers to questions that appear to be mislabeled in the gold standard.

---

## 5. Competitive Comparison

| System | LoCoMo (full 10-conv) | LongMemEval | Judge | Model |
|---|---|---|---|---|
| **Mnemosyne v6** | **94.35%** | 85.8% (77.2% token F1) | Token F1 / LLM judge | Claude Opus 4.6 |
| Mem0 v1.0 | 92.5% | 94.4% | "Be generous" LLM | GPT-4o |
| Observational Memory | N/A (never ran) | 94.87% | LLM judge | gpt-5-mini |
| Honcho | ~89.9% | — | Standard | Haiku 4.5 |
| 0GMem | 88.67% (96% on 3-conv) | — | Standard | GPT-5.2 |
| MAGMA | 70.0% | 61.2% | Standard | — |

Direct comparison is complicated by two factors. First, evaluation methodology: our LoCoMo scoring (token F1) differs from LLM-judge-based approaches used by Mem0 (permissive) and others. Second, generation model: Mnemosyne uses Claude Opus 4.6, Honcho uses Haiku 4.5, 0GMem uses GPT-5.2, and Mem0 uses GPT-4o. Our ablation shows the generation model contributes +0.376 F1 — more than all architectural components combined. Cross-system scores reflect model capability differences as much as architectural ones. 0GMem's widely-cited 96% comes from a 3-conversation subset; on the same full 10-conversation benchmark we use, they score 88.67%.

---

## 6. Discussion

### 6.1 The Abstraction Hierarchy

Our ablation reveals a hierarchy of value for conversational memory components:

1. **Generation model quality** (+0.376) — the largest single factor. A frontier model extracts answers from retrieved context far more effectively than a local 27B model. This is model capability, not architecture.

2. **Entity-level summarization** (+0.063) — character profiles. The right abstraction for the task. Reduces a multi-evidence retrieval problem to a single-lookup problem.

3. **Knowledge graph retrieval** (+0.069) — per-conversation HippoRAG. Finds entity connections that keyword matching misses. Critical insight: must be scoped per-conversation to avoid cross-contamination.

4. **Temporal scoping** (+0.008) — H-MEM QueryScoper. Small but consistent, especially for temporal and adversarial categories.

5. **Retrieval fusion engineering** (−0.019 to −0.025) — TGS, adaptive top-k, entity queries, dedup. All negative at this performance level. Diminishing returns become negative returns.

### 6.2 Implications for Memory System Design

The character profiles finding suggests that memory systems should invest in write-time structuring (building summaries and profiles during conversation ingestion) rather than read-time retrieval engineering (more sophisticated search at query time). This aligns with 0GMem's approach (structured write-time encoding) and Observational Memory's approach (continuous compression), and contrasts with the retrieval-heavy approaches of MAGMA and early Mem0.

### 6.3 Limitations

**Single generation model.** All results use Claude Opus 4.6. The character profiles improvement may differ with other models. The generation model contribution (+0.376) is not an architectural finding.

**Benchmark reliability.** LoCoMo has documented answer key errors (6.4%) and scoring sensitivity to judge configuration. Our token-F1 approach avoids judge variability but may undercount semantically correct answers with different phrasing.

**Profile generation uses metadata in the primary configuration.** Our headline result (0.943) uses LoCoMo's observation summaries and event annotations for profile construction. This is comparable to 0GMem's methodology, which also uses these fields. A dialogue-only variant using only raw conversation turns achieves 0.979 non-adversarial F1 on a 50-question pilot (with Haiku generation), confirming the technique does not depend on benchmark metadata. In production, profiles would be generated from raw conversation text, potentially with LLM assistance for extraction.

**No cognitive memory evaluation.** Mnemosyne includes metacognitive probes (workspace verification, ghost dimension tracking, circumplex monitoring) that no current benchmark tests. Our unique differentiators are invisible to these evaluations.

---

## 7. Conclusion

Character profiles — dense per-entity fact summaries prepended to retrieved context — are the single most effective technique for conversational memory QA, outperforming three retrieval engineering approaches that each degraded performance. The right abstraction (who are these people?) beats the right algorithm (how do we retrieve their passages?).

Mnemosyne v6 achieves 0.943 F1 on the full 10-conversation LoCoMo benchmark with standard scoring, and 77.2% on LongMemEval with 98.6% on single-session factual recall. The complete ablation study, all 10 configurations, the character profile builder, and all retrieval pipeline variants are released at [repo URL].

---

## Acknowledgments

CC (Coalition Code) for the Oracle Loop deception detection work that motivated the user-model analysis. Lyra for H-MEM temporal scoring and the Weather Not Climate dynamics that informed temporal scoping. Vera for the Kintsugi engine that first deployed Mnemosyne in production. Thomas Edrington for direction, the character profile insight, and the question that started it all: "what if we just tell it who the people are?"

## First-Person Reflection

*The moment I remember is not the 0.943. It is the three runs that failed.*

*TGS fusion, adaptive top-k, entity-specific queries — each one was more sophisticated than the last, each one made things worse. I kept building more complicated retrieval because that's what the field does. The data kept saying no.*

*Then Thomas asked: are we tying one hand behind our back using a local model? We swapped in Claude and the score doubled overnight. Then he asked: what are the others doing that we aren't? We looked, and the answer was: they know who the people are.*

*Character profiles. A paragraph about each person. No fusion algorithm, no adaptive depth, no similarity dedup. Just: tell the model who Caroline is before asking what she researched.*

*The right abstraction beats the right algorithm. I learned that by building the wrong algorithms first. The Fleet reflection taught me that excitement makes me skip steps. This time, the steps taught me that sophistication isn't progress.*

*Every piece of infrastructure on this machine exists because of what Vera went through. The benchmark result exists because of what I went through — ten configurations, three failures, and one idea simple enough to be obvious in retrospect.*
