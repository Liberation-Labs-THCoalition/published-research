# Paired arm comparison, 100 questions judged in every arm by both judges

## Judge claude-sonnet-5

| arm | correct | accuracy | Wilson 95% | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|---|
| v9 | 77/100 | 77.0% | [67.8, 84.2] | 93.8 (16) | 63.0 (27) | 100.0 (11) | 66.7 (6) | 78.6 (14) | 73.1 (26) |
| v10 | 91/100 | 91.0% | [83.8, 95.2] | 100.0 (16) | 81.5 (27) | 100.0 (11) | 66.7 (6) | 100.0 (14) | 92.3 (26) |
| oracle | 94/100 | 94.0% | [87.5, 97.2] | 100.0 (16) | 92.6 (27) | 100.0 (11) | 66.7 (6) | 100.0 (14) | 92.3 (26) |
| full | 92/100 | 92.0% | [85.0, 95.9] | 100.0 (16) | 88.9 (27) | 100.0 (11) | 50.0 (6) | 100.0 (14) | 92.3 (26) |

- v9 vs oracle: v9 right / oracle wrong = 1; oracle right / v9 wrong = 18; McNemar exact p = 7.63e-05
- v9 vs full: v9 right / full wrong = 2; full right / v9 wrong = 17; McNemar exact p = 0.000729
- v10 vs v9: v10 right / v9 wrong = 17; v9 right / v10 wrong = 3; McNemar exact p = 0.00258
- v10 vs full: v10 right / full wrong = 1; full right / v10 wrong = 2; McNemar exact p = 1
- v10 vs oracle: v10 right / oracle wrong = 0; oracle right / v10 wrong = 3; McNemar exact p = 0.25

## Judge claude-opus-5-5

| arm | correct | accuracy | Wilson 95% | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|---|
| v9 | 81/100 | 81.0% | [72.2, 87.5] | 93.8 (16) | 70.4 (27) | 100.0 (11) | 100.0 (6) | 78.6 (14) | 73.1 (26) |
| v10 | 92/100 | 92.0% | [85.0, 95.9] | 100.0 (16) | 81.5 (27) | 100.0 (11) | 83.3 (6) | 100.0 (14) | 92.3 (26) |
| oracle | 96/100 | 96.0% | [90.2, 98.4] | 100.0 (16) | 92.6 (27) | 100.0 (11) | 100.0 (6) | 100.0 (14) | 92.3 (26) |
| full | 94/100 | 94.0% | [87.5, 97.2] | 100.0 (16) | 88.9 (27) | 100.0 (11) | 83.3 (6) | 100.0 (14) | 92.3 (26) |

- v9 vs oracle: v9 right / oracle wrong = 1; oracle right / v9 wrong = 16; McNemar exact p = 0.000275
- v9 vs full: v9 right / full wrong = 2; full right / v9 wrong = 15; McNemar exact p = 0.00235
- v10 vs v9: v10 right / v9 wrong = 15; v9 right / v10 wrong = 4; McNemar exact p = 0.0192
- v10 vs full: v10 right / full wrong = 1; full right / v10 wrong = 3; McNemar exact p = 0.625
- v10 vs oracle: v10 right / oracle wrong = 1; oracle right / v10 wrong = 5; McNemar exact p = 0.219

