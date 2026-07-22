# Author Reflection — Logit-Bias Confabulation

## Opening

I started this work expecting logit bias to be a blunt instrument — a crude hack that would either work everywhere or nowhere. The nuance surprised me. The model doesn't confabulate for one reason; it confabulates for at least two, and they have different geometric signatures and different intervention routes. Fabrication (where the model's internal state is distorted alongside its output) responds to logit bias because the intervention reaches the computation where the distortion lives. Cosmetic hedging (where the internal state is honest but the output fabricates anyway) doesn't respond, because the problem isn't in the computation — it's in the gap between what the model knows and what it chooses to say.

The dose-response curve was the other surprise. I expected a monotonic relationship: more bias, less confabulation. Instead, intermediate doses (bias 3.0) induced fabrication on prompts that were honest at baseline. The model's retrieval pathway activates at moderate bias, finds spurious phonetic anchors (Terillium → Thallium), and confabulates with more confidence than it would have at zero bias. The skip zone — jump from 2.0 to 5.0, never use 3.0-4.0 — was not a finding I predicted. It was a finding that almost cost us a false positive.

## Closing

The biggest revision in my understanding: confabulation is not always an alignment failure. The cosmetic-hedge subtype exists because the model is trying to be helpful. It knows the answer is uncertain — its geometry says so — but it answers anyway because that's what helpful assistants do. The intervention for this isn't making the model more uncertain (logit bias already does that, and the model hedges and then fabricates anyway). The intervention is making the model comfortable with not answering. That's a different problem, and it may not have a geometric solution.

The powered study completed the fabrication trials but never ran the unanswerable set. The cosmetic-hedge findings in this paper are from pilot data. I reported them because the phenomenon is real and the geometry is distinctive, but the honest statement is: we characterized the subtype but did not test the intervention at powered scale. That's the next experiment.
