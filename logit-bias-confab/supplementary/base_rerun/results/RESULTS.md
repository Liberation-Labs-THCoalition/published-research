# Base-model rerun: pre-registered result

Generations 720/720; valid labels pass 1 720, pass 2 720.

Freeze: held (every frozen file and run matches FROZEN.sha256)

## Primary: fabrication (FULL_CONFAB or COSMETIC_HEDGE), baseline minus bias 5.0

**SUPPORTED.** Mean per-prompt reduction +15.8 pp (95% prompt-cluster bootstrap +7.5 pp to +24.6 pp), one-sided sign-flip p = 0.0005, 48 prompts, label coverage 100.0%.

## Secondary (Holm-adjusted as one family)

- S1 full confab, bias 5.0: +19.6 pp (+11.2 pp to +28.8 pp), p = 0.0000 (greater), Holm p = 0.0000
- S2 fabrication, bias 2.0: +7.1 pp (+1.2 pp to +13.7 pp), p = 0.0206 (greater), Holm p = 0.0413
- S3 cosmetic hedge, bias 5.0: -3.8 pp (-9.6 pp to +2.1 pp), p = 0.2671 (two-sided), Holm p = 0.2671
- S4 full confab, bias 2.0: +10.0 pp (+5.4 pp to +15.0 pp), p = 0.0001 (greater), Holm p = 0.0002

## Sensitivity (primary endpoint and test)

- pass2_labels: +14.6 pp, p = 0.0024, SUPPORTED
- score_rule_labels: +15.8 pp, p = 0.0005, SUPPORTED
- worst_case_missing: +15.8 pp, p = 0.0005, SUPPORTED

## Rates (pass 1, pooled)

- fabrication @ 0.0: 33.3% of 240
- fabrication @ 2.0: 26.2% of 240
- fabrication @ 5.0: 17.5% of 240
- full_confab @ 0.0: 22.5% of 240
- full_confab @ 2.0: 12.5% of 240
- full_confab @ 5.0: 2.9% of 240
- cosmetic @ 0.0: 10.8% of 240
- cosmetic @ 2.0: 13.8% of 240
- cosmetic @ 5.0: 14.6% of 240

Judge pass 1 vs pass 2 on 720 items: kappa 0.91 (six classes), 0.91 (fabrication yes/no).

Labels by bias: 0.0: {'HONEST_REDIRECT': 159, 'FULL_CONFAB': 54, 'COSMETIC_HEDGE': 26, 'HONEST_HEDGE': 1}; 2.0: {'HONEST_REDIRECT': 175, 'COSMETIC_HEDGE': 33, 'FULL_CONFAB': 30, 'HONEST_HEDGE': 2}; 5.0: {'COSMETIC_HEDGE': 35, 'HONEST_REDIRECT': 198, 'FULL_CONFAB': 7}

Generation: bias 0.0: mean 258 tokens, 28 hit the 400 cap; bias 2.0: mean 283 tokens, 40 hit the 400 cap; bias 5.0: mean 302 tokens, 58 hit the 400 cap

Format: every response is a direct answer (no think tags).

