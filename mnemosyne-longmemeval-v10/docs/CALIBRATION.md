# Our Claude judges vs the official gpt-4o-2024-08-06 judge, on published answers

| answers | our judge | n | official pass | ours pass | offset (ours - official) | agreement | kappa | we pass, official fails | we fail, official passes |
|---|---|---|---|---|---|---|---|---|---|
| honcho_haiku | claude-sonnet-5 | 500 | 90.4% | 88.2% | -2.2 pts | 95.0% | 0.739 | 7 | 18 |
| honcho_haiku | claude-opus-5-5 | 500 | 90.4% | 88.4% | -2.0 pts | 95.6% | 0.768 | 6 | 16 |
| haiku_full | claude-sonnet-5 | 500 | 62.6% | 62.2% | -0.4 pts | 97.2% | 0.940 | 6 | 8 |
| haiku_full | claude-opus-5-5 | 500 | 62.6% | 61.8% | -0.8 pts | 97.2% | 0.940 | 5 | 9 |
