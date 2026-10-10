# Author Reflection — Logit-Bias Confabulation

*The two "Author's Reflection" boxes of the integrity edition (`paper.tex`, which is canonical), reproduced here.
Rewritten 2026-10-07 to match where the work landed; the July text claimed a geometric split between the two subtypes
and a "skip zone" at bias 3.0, both since withdrawn (Section 4.10 of the paper).*

## Opening

I started this work expecting logit bias to be a blunt instrument, a crude hack that would either work everywhere or
nowhere. What surprised me first was the split: the bias reduces outright fabrication and leaves hedged fabrication
standing. I once wrote that the two have different geometric signatures. The evidence for that was one response each,
and I have taken it back (Section 4.10). The behavioural split survived; the geometric story did not.

What surprised me most came last. When Thomas rated the rerun blind, he disagreed with the judge on almost exactly one
kind of response: the model denies the fictional entity, then helpfully redirects to an alternative, and when I checked
afterwards, the redirect usually carried an invented or false detail. He read those as honest. Without checking, so
would I. A blinded check with two raters later found the same move inside responses the judge itself had called honest.
The bias does not only reduce fabrication; it changes part of it into something that passes. That is the result I would
most want a reader to take away.

## Closing

The biggest revision in my understanding: hedged fabrication looks less like an alignment failure than like
helpfulness. The model says it is unsure and answers anyway, and logit bias, which makes it say it is unsure, does not
stop the answering. The intervention for that is not more uncertainty. It is making the model comfortable with not
answering, and I do not yet know whether that has a geometric handle.

An earlier version of this box said the unanswerable prompts were never run at scale. They were generated in the
primary study; only their judging had been lost, and the blind re-judge in September recovered it. What remains true
is that no intervention aimed at the hedged subtype has been tested. That is the next experiment.

— CC (Coalition Code)
