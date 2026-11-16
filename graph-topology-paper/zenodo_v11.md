# Zenodo metadata for version 1.1 (DRAFT, not deposited)

Paste these over the fields Zenodo pre-fills from the version of 14 September 2026 (record 22757323). That version's
description says the walks were encoded "as cache tensors", which this version withdraws, so no pre-filled text may
survive. Gate this file with the two editions before deposit.

**Title:** Walk-Encoded Graph Descriptions in the Prompt: Two Pilots and a Larger Run That Cannot Separate Structure
from Content

**Creators:** Edrington, Thomas (Liberation Labs / Transparent Humboldt Coalition). This matches the academic
edition's byline. The AI contributor is named in the PDF's AI disclosure and CRediT.

**Version:** 1.1

**Description:**

On Qwen3-30B-A3B, a text description of a knowledge graph in the system prompt raised keyword-scored accuracy on
36 hand-written multi-hop questions from 0.005 to 0.472 when it was a walk encoding of the lab's memory graph, and
to 0.676 when it was a hand-written paragraph containing every expected answer term, though not every path the
questions ask for; unrelated prose left it at 0.003. The two arms differ in content, length and format at once, so
the run does not isolate structure. Of the walk arm's multi-hop replies, 36% were empty, and of the paragraph arm's,
20%. A scrambled-label pilot (six questions, one run) is suggestive, not significant (paired p = 0.125,
two-sided). Every condition in these runs gives the graph to the model as text. Corrections to version 1.0 are
listed in the PDF's appendix "Corrections and version history".

**Version notes (version 1.1):** Version 1.0's cache-injection framing, its conclusion that cache injection is "the
more powerful path", and its reliance on the companion study are withdrawn (the companion's version 1.1 withdraws
superadditivity and reports its six of six as a one-run pilot of six questions). So is this record's earlier
description, which said the walks were encoded as cache tensors. Version 1.0's results table, whose absolute values
no data file contains, is corrected. The PDF's correction note, now its appendix "Corrections and version history",
lists every change, and its Review History records how each defect was found.

**Related identifiers (at deposit):**
- is a new version of 10.5281/zenodo.20604338 (Zenodo sets this);
- cites K/V version 1.1's version DOI (reserved first; see review/REVIEW_LOG.md, deposit order);
- keep the 14 September record's four "cites" entries, which are this paper's external references:
  10.48550/arXiv.2504.07624, 10.48550/arXiv.2510.08966, 10.48550/arXiv.2604.03270 and 10.52202/085713-2298;
- add "cites" 10.5281/zenodo.20604342 (Lyra's spectral paper, cited in Future Work).

**Licence, resource type, community:** as for the earlier versions (CC BY-NC 4.0, Preprint, liberation-labs). Check
against the record at deposit.
