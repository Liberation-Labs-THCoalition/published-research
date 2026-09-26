# Paired arm comparison, 400 questions judged in every arm by both judges (v10 split: held)

## Judge claude-sonnet-5

| arm | correct | accuracy | Wilson 95% | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|---|
| v9 | 316/400 | 79.0% | [74.7, 82.7] | 88.7 (62) | 56.6 (106) | 100.0 (45) | 66.7 (24) | 100.0 (56) | 78.5 (107) |
| v10 | 369/400 | 92.2% | [89.2, 94.5] | 93.5 (62) | 87.7 (106) | 97.8 (45) | 66.7 (24) | 100.0 (56) | 95.3 (107) |
| v10_o55t | 385/400 | 96.2% | [93.9, 97.7] | 98.4 (62) | 94.3 (106) | 97.8 (45) | 91.7 (24) | 100.0 (56) | 95.3 (107) |
| oracle | 381/400 | 95.2% | [92.7, 96.9] | 95.2 (62) | 92.5 (106) | 100.0 (45) | 79.2 (24) | 100.0 (56) | 97.2 (107) |
| oracle_o55t | 392/400 | 98.0% | [96.1, 99.0] | 98.4 (62) | 96.2 (106) | 100.0 (45) | 87.5 (24) | 100.0 (56) | 100.0 (107) |

- v9 vs oracle: v9 right / oracle wrong = 8; oracle right / v9 wrong = 73; McNemar exact p = 2.98e-14
- v10 vs v9: v10 right / v9 wrong = 65; v9 right / v10 wrong = 12; McNemar exact p = 5.92e-10
- v10 vs oracle: v10 right / oracle wrong = 6; oracle right / v10 wrong = 18; McNemar exact p = 0.0227
- v10_o55t vs v10: v10_o55t right / v10 wrong = 19; v10 right / v10_o55t wrong = 3; McNemar exact p = 0.000855
- oracle_o55t vs oracle: oracle_o55t right / oracle wrong = 14; oracle right / oracle_o55t wrong = 3; McNemar exact p = 0.0127
- v10_o55t vs oracle_o55t: v10_o55t right / oracle_o55t wrong = 5; oracle_o55t right / v10_o55t wrong = 12; McNemar exact p = 0.143

## Judge claude-opus-5-5

| arm | correct | accuracy | Wilson 95% | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|---|
| v9 | 320/400 | 80.0% | [75.8, 83.6] | 88.7 (62) | 57.5 (106) | 97.8 (45) | 79.2 (24) | 100.0 (56) | 79.4 (107) |
| v10 | 378/400 | 94.5% | [91.8, 96.3] | 93.5 (62) | 89.6 (106) | 97.8 (45) | 91.7 (24) | 100.0 (56) | 96.3 (107) |
| v10_o55t | 387/400 | 96.8% | [94.5, 98.1] | 98.4 (62) | 95.3 (106) | 97.8 (45) | 91.7 (24) | 100.0 (56) | 96.3 (107) |
| oracle | 387/400 | 96.8% | [94.5, 98.1] | 91.9 (62) | 94.3 (106) | 100.0 (45) | 100.0 (24) | 100.0 (56) | 98.1 (107) |
| oracle_o55t | 397/400 | 99.2% | [97.8, 99.7] | 100.0 (62) | 98.1 (106) | 97.8 (45) | 100.0 (24) | 100.0 (56) | 100.0 (107) |

- v9 vs oracle: v9 right / oracle wrong = 5; oracle right / v9 wrong = 72; McNemar exact p = 2.8e-16
- v10 vs v9: v10 right / v9 wrong = 64; v9 right / v10 wrong = 6; McNemar exact p = 2.44e-13
- v10 vs oracle: v10 right / oracle wrong = 6; oracle right / v10 wrong = 15; McNemar exact p = 0.0784
- v10_o55t vs v10: v10_o55t right / v10 wrong = 10; v10 right / v10_o55t wrong = 1; McNemar exact p = 0.0117
- oracle_o55t vs oracle: oracle_o55t right / oracle wrong = 12; oracle right / oracle_o55t wrong = 2; McNemar exact p = 0.0129
- v10_o55t vs oracle_o55t: v10_o55t right / oracle_o55t wrong = 1; oracle_o55t right / v10_o55t wrong = 11; McNemar exact p = 0.00635

