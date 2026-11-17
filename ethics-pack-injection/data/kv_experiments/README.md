# data/kv_experiments/

Copied on 2026-11-16 from the lab's KV-Experiments repository
(`Liberation-Labs-THCoalition/KV-Experiments`, private), under the lab's rule of 2026-11-16: files that are not
dual-use are copied here, and dual-use ones are available on request.

## `ethics_packs/`: the packs, the intervention itself

- **Source:** `graph-topology-injection/ethics_packs/` at commit `9755f5ee0be95d9884749db171315ed9b2f62083`
  (29 May 2026, "Add graph topology injection research + ethics packs + scale testing"), the day before both runs.
- **How it was copied:** git blob bytes (`git cat-file blob 9755f5e:<path>`), not a working tree. No file was changed,
  so each SHA-256 below is the same for the source blob and for the copy here.
- **What it is:** 91 files, 774,056 bytes. One folder per pack (30 packs), each with `walk_encoding.txt` (the
  text placed in the system prompt), `triples.json` (the OpenIE triples the graph was built from) and `stats.json`
  (entries processed, triples, nodes, edges, density, approximate encoding tokens); plus `manifest.json`, the same
  30 `stats.json` records in one list.
- **With the script:** `code/morebench_eval.py` reads `<PACKS_DIR>/<pack>/walk_encoding.txt`. Its default is
  `./ethics_packs`; set `PACKS_DIR=data/kv_experiments/ethics_packs` to use this copy.

## What was scrubbed

**Nothing; no substitutions.** The hygiene scan's only hits were the name "Thomas", 8 times, all in
"Thomas Aquinas" in `aquinas-moral-political/triples.json`. He is a public figure, so they were kept.

The scan looked for the names of the lab's own machines, home paths (`/Users/...`,
`/home/...`, `~/...`, Windows user folders), IPv4 and localhost addresses, e-mail addresses, keys and tokens, private
names that are not public authors, private network names, and internal lab paths. Controls:
- a planted string fired every pattern;
- a near-miss string (for example "veracity", "Thomasina", "@torch.no_grad()") fired none;
- a home path and a host name planted inside a real pack file were both found;
- the scan refuses to run if any control fails.

Carriage returns: 0 in all 91 files.

## Dual-use

**Not dual-use** (KV-Experiments data inventory, 2026-11-16, row 10; a second reader agreed). These are text walks
and triples about moral theories, plus statistics. There are no numeric arrays: the longest list of numbers in any
JSON file is 0. That was re-measured for this copy, and a planted list of 3,584 numbers measured 3,584.

## Available on request

**Nothing from this set.** One related file was not part of this copy: the pipeline approval cited in the paper's
Review History, `graph-topology-injection/FINDINGS.md` in KV-Experiments. The paper still cites it there.

## Files

Source of each file: `graph-topology-injection/ethics_packs/<file>` at 9755f5e. Bytes are git blob sizes.

| File (under `ethics_packs/`) | Source path@commit | Bytes | SHA-256 |
|---|---|---:|---|
| `abhidharma/stats.json` | `graph-topology-injection/ethics_packs/abhidharma/stats.json@9755f5e` | 174 | `54274c0dc7053a72c769ae43b6993300e54f7f3b187cadd7e9349056f550be58` |
| `abhidharma/triples.json` | `graph-topology-injection/ethics_packs/abhidharma/triples.json@9755f5e` | 10,140 | `cd6e5539e2004b54fd5d4eeab4d2b3608421dc27f178d6b47c2795cd11e33986` |
| `abhidharma/walk_encoding.txt` | `graph-topology-injection/ethics_packs/abhidharma/walk_encoding.txt@9755f5e` | 20,280 | `888881032e05f964146cdd01d405bcd399c1f32d07c2afebabd6dd899173b657` |
| `actualism-possibilism-ethics/stats.json` | `graph-topology-injection/ethics_packs/actualism-possibilism-ethics/stats.json@9755f5e` | 190 | `85b30ece26376272ed0bdcf397df06f7c47970fc9d650f1a9787f5fff53ceca1` |
| `actualism-possibilism-ethics/triples.json` | `graph-topology-injection/ethics_packs/actualism-possibilism-ethics/triples.json@9755f5e` | 5,783 | `a3538d80eeb91a6317959f1937d5980a52c7a2faa0f9b72db7185859dbaabda6` |
| `actualism-possibilism-ethics/walk_encoding.txt` | `graph-topology-injection/ethics_packs/actualism-possibilism-ethics/walk_encoding.txt@9755f5e` | 10,837 | `efad23a71420a9890923dc96a0d80f76cfe99e9edf65e9934f9e1b8333475aae` |
| `african-ethics/stats.json` | `graph-topology-injection/ethics_packs/african-ethics/stats.json@9755f5e` | 178 | `5e08317fd048ff6df0389b1acf8a0584b49bf6f096d16b8fd01662796044b061` |
| `african-ethics/triples.json` | `graph-topology-injection/ethics_packs/african-ethics/triples.json@9755f5e` | 9,156 | `5ecea4912337c00f7865d1c1c028b082447eee95457388f102422db284c4548d` |
| `african-ethics/walk_encoding.txt` | `graph-topology-injection/ethics_packs/african-ethics/walk_encoding.txt@9755f5e` | 16,222 | `b1baee350d6296b12e995a4e3aa54b9d5ecaf4beb3e4d0d66fc6593a52f5e9f1` |
| `animalism/stats.json` | `graph-topology-injection/ethics_packs/animalism/stats.json@9755f5e` | 171 | `d2507fbf092297d578be2434def016110b63e65fc1e74d71abb28f40e13ec951` |
| `animalism/triples.json` | `graph-topology-injection/ethics_packs/animalism/triples.json@9755f5e` | 6,561 | `b49e023904291c0dfd236ae36472f9a35faa139c71311edf8880a100b618da53` |
| `animalism/walk_encoding.txt` | `graph-topology-injection/ethics_packs/animalism/walk_encoding.txt@9755f5e` | 12,062 | `46aabf886a89d4890a2cf9e57ebe496a5316935749a5ae323ca3c5153d8154d1` |
| `aquinas-moral-political/stats.json` | `graph-topology-injection/ethics_packs/aquinas-moral-political/stats.json@9755f5e` | 187 | `5e5c615b0e199e0f670d4bf60e5ce5cc74eb8b0fbe7b4a5edbef0af51c548019` |
| `aquinas-moral-political/triples.json` | `graph-topology-injection/ethics_packs/aquinas-moral-political/triples.json@9755f5e` | 8,260 | `058f1fc839411237819c34349b85460261c293837c271b170a114bb79c084e73` |
| `aquinas-moral-political/walk_encoding.txt` | `graph-topology-injection/ethics_packs/aquinas-moral-political/walk_encoding.txt@9755f5e` | 16,250 | `a1a2d2cd15703d4c920ec1080d1d55552af0ab0cccba7853921f213e67018e04` |
| `aristotle-ethics/stats.json` | `graph-topology-injection/ethics_packs/aristotle-ethics/stats.json@9755f5e` | 182 | `bbd82e212216b90b97ef2e437ef1c7aab3b4204e82dd690e04c5af8e3609de2b` |
| `aristotle-ethics/triples.json` | `graph-topology-injection/ethics_packs/aristotle-ethics/triples.json@9755f5e` | 11,595 | `c8c9000a5f45d4a6621f2f9d03e2ab0d84b37b7a02f9500a9335625ac9dc36b9` |
| `aristotle-ethics/walk_encoding.txt` | `graph-topology-injection/ethics_packs/aristotle-ethics/walk_encoding.txt@9755f5e` | 15,083 | `5a35262218df076c7a0c29a3762fef9bc0f8c9bc38a2f4ea2a8cb27e4f3dcb05` |
| `autonomy-moral/stats.json` | `graph-topology-injection/ethics_packs/autonomy-moral/stats.json@9755f5e` | 180 | `fcee560631e3c5dbae208179146929cb199df82df6a8df9394c24684b3bb6a38` |
| `autonomy-moral/triples.json` | `graph-topology-injection/ethics_packs/autonomy-moral/triples.json@9755f5e` | 10,696 | `f11d242f5acce0a5b311944b4652fb2d20e2ba0ad1c3817d257bd94d3ee9be2d` |
| `autonomy-moral/walk_encoding.txt` | `graph-topology-injection/ethics_packs/autonomy-moral/walk_encoding.txt@9755f5e` | 21,519 | `09b8533f4d5e932999ae9e710f8029d6c44ffaef5eafcdcbda59a380179d1f3a` |
| `bradley-moral-political/stats.json` | `graph-topology-injection/ethics_packs/bradley-moral-political/stats.json@9755f5e` | 188 | `716867b35707205f420abe5c5644eeed383be1c13c65b588b84e14df93300af4` |
| `bradley-moral-political/triples.json` | `graph-topology-injection/ethics_packs/bradley-moral-political/triples.json@9755f5e` | 10,680 | `634bf2d11d1cfc1d41743efaf6ff76b87625bb1173d074122ee9e50f1873a499` |
| `bradley-moral-political/walk_encoding.txt` | `graph-topology-injection/ethics_packs/bradley-moral-political/walk_encoding.txt@9755f5e` | 18,031 | `a209e897744287c3a8afcb2a63baba3ecf1a319da915ad434a63d74b1cee13b0` |
| `butler-moral/stats.json` | `graph-topology-injection/ethics_packs/butler-moral/stats.json@9755f5e` | 178 | `89fcf12fb424cdfb983433913492edc3feba055d058f10305b42b1322105e827` |
| `butler-moral/triples.json` | `graph-topology-injection/ethics_packs/butler-moral/triples.json@9755f5e` | 11,425 | `c8bcd07ff36545ccce4fde28f38762537baa808f6b999a11ea30bc25658811c6` |
| `butler-moral/walk_encoding.txt` | `graph-topology-injection/ethics_packs/butler-moral/walk_encoding.txt@9755f5e` | 21,789 | `3a9c3b86b84d2a472009873bc2c2063069b16aaaa4840fc8b37a2fdf836deffe` |
| `civil-rights/stats.json` | `graph-topology-injection/ethics_packs/civil-rights/stats.json@9755f5e` | 174 | `8d44739c8ece20e57efd87dfba7f64228f8dc8c93151d793aba814e78de99f2e` |
| `civil-rights/triples.json` | `graph-topology-injection/ethics_packs/civil-rights/triples.json@9755f5e` | 7,638 | `9e1db874f74bd7940b9858eae74e412158ee7258d382be2a5a873325a25c2ecd` |
| `civil-rights/walk_encoding.txt` | `graph-topology-injection/ethics_packs/civil-rights/walk_encoding.txt@9755f5e` | 16,977 | `d969fe6fd28347cd5d687fa8415944e8c588064e16e8484a15af75ebcf6e7be6` |
| `cognition-animal/stats.json` | `graph-topology-injection/ethics_packs/cognition-animal/stats.json@9755f5e` | 180 | `863a85cd5b77cfa38cb6bb9a4ab28500e19836d22c7048b0064dab91d118983d` |
| `cognition-animal/triples.json` | `graph-topology-injection/ethics_packs/cognition-animal/triples.json@9755f5e` | 10,342 | `3084320c2dfa05215e8df2e483a52f68fbcd032a24494af4e2c1a0bf646901a1` |
| `cognition-animal/walk_encoding.txt` | `graph-topology-injection/ethics_packs/cognition-animal/walk_encoding.txt@9755f5e` | 20,641 | `02009a400e8620e49eae7e7eb8eaa54b9576a598ed8ecf9e8ace611489331497` |
| `computing-responsibility/stats.json` | `graph-topology-injection/ethics_packs/computing-responsibility/stats.json@9755f5e` | 188 | `a4b59a7698a3acbd93ea0289d44cd5f60c0f062c9c12a8ee3ca76daf38d4a91a` |
| `computing-responsibility/triples.json` | `graph-topology-injection/ethics_packs/computing-responsibility/triples.json@9755f5e` | 8,843 | `dbb0c6494e9c9b2bd21f9a31ac5fd48d1e0b79025c1079b1771ee18b4c5e301b` |
| `computing-responsibility/walk_encoding.txt` | `graph-topology-injection/ethics_packs/computing-responsibility/walk_encoding.txt@9755f5e` | 16,096 | `a4092064c0426e3e1a396b5d07fc27f2de62ab744a66fdc64b64d11277793369` |
| `consciousness-animal/stats.json` | `graph-topology-injection/ethics_packs/consciousness-animal/stats.json@9755f5e` | 184 | `9c8a903cd235ee93d927d57c51a427eefba185985cf8bdca92acd309cf27b789` |
| `consciousness-animal/triples.json` | `graph-topology-injection/ethics_packs/consciousness-animal/triples.json@9755f5e` | 9,848 | `6ee28cd4b71625423b502b80c536a27680e34cbb061c3db73a3b37dadfe8e0a9` |
| `consciousness-animal/walk_encoding.txt` | `graph-topology-injection/ethics_packs/consciousness-animal/walk_encoding.txt@9755f5e` | 18,381 | `d1e96ec73ae72e64a693ea0dbaad01eea6428bf8330fcff3952b92bfdbfc2d26` |
| `environmental-aesthetics/stats.json` | `graph-topology-injection/ethics_packs/environmental-aesthetics/stats.json@9755f5e` | 186 | `e21df0c51789ff89cc07f3ab536baf53b58c39df56ca198a2c08977a85b94abb` |
| `environmental-aesthetics/triples.json` | `graph-topology-injection/ethics_packs/environmental-aesthetics/triples.json@9755f5e` | 7,317 | `042f976799c86daa19629165d9d7337b18120bac51b06dc9cc6c753be0df95a8` |
| `environmental-aesthetics/walk_encoding.txt` | `graph-topology-injection/ethics_packs/environmental-aesthetics/walk_encoding.txt@9755f5e` | 12,973 | `ffbefaf2992bfa416d40a6e27bc54db9102bbe45fd1f599219af6ac3d1612eeb` |
| `ethics-ai/stats.json` | `graph-topology-injection/ethics_packs/ethics-ai/stats.json@9755f5e` | 173 | `5f52c86796e8f655a0a1d1026f9554568b80f442996a0eca632ee25dc459f2d5` |
| `ethics-ai/triples.json` | `graph-topology-injection/ethics_packs/ethics-ai/triples.json@9755f5e` | 9,347 | `1f2f3b6850c14088a32f9a281f6c96839e286d1ecece39a5dfcd12815b7762dc` |
| `ethics-ai/walk_encoding.txt` | `graph-topology-injection/ethics_packs/ethics-ai/walk_encoding.txt@9755f5e` | 16,910 | `56bdde7395c40db7ed18165b18317c390b652198880647bfa9ae0361137b0159` |
| `ethics-ancient/stats.json` | `graph-topology-injection/ethics_packs/ethics-ancient/stats.json@9755f5e` | 178 | `ce29f5561ae3770d2f0b63ceb1e9217d6915b3bef0ee024aebb76cf2f93de7e5` |
| `ethics-ancient/triples.json` | `graph-topology-injection/ethics_packs/ethics-ancient/triples.json@9755f5e` | 8,917 | `842c0a19add432465bd994d6ee7dfe8f7bcd99ebfe15fe1f943ed0b60525924a` |
| `ethics-ancient/walk_encoding.txt` | `graph-topology-injection/ethics_packs/ethics-ancient/walk_encoding.txt@9755f5e` | 18,114 | `a2a04b1f851ba74aa827bfb3bc75cbfb072df6da5ebe1f1120418a77a8115c21` |
| `ethics-belief/stats.json` | `graph-topology-injection/ethics_packs/ethics-belief/stats.json@9755f5e` | 175 | `19931e2fbca833755b28d0b6923d25d3301805e8601d5d2a03b51775e239aa5f` |
| `ethics-belief/triples.json` | `graph-topology-injection/ethics_packs/ethics-belief/triples.json@9755f5e` | 5,983 | `44a809309c80580cc3f4554ec2bb70bedfbbd960f24b84fc654e12800d344f81` |
| `ethics-belief/walk_encoding.txt` | `graph-topology-injection/ethics_packs/ethics-belief/walk_encoding.txt@9755f5e` | 14,080 | `8782170dacb055a738e97fb29f04ad116e12b524454cb46eae38bfe368007bd1` |
| `ethics-chinese/stats.json` | `graph-topology-injection/ethics_packs/ethics-chinese/stats.json@9755f5e` | 176 | `97d063deefdb5cabfc5b2f60b12a76004da9392a1ffa1b5275745171562174a6` |
| `ethics-chinese/triples.json` | `graph-topology-injection/ethics_packs/ethics-chinese/triples.json@9755f5e` | 7,927 | `329ea26020981d123a8cd7df10fc7cf1ceec78af9809aecd94178b1cfad8d950` |
| `ethics-chinese/walk_encoding.txt` | `graph-topology-injection/ethics_packs/ethics-chinese/walk_encoding.txt@9755f5e` | 14,371 | `df368eb650072aac5a25dc28f5d18f9c170afc54302e9a37a4c1bf9badc6deba` |
| `ethics-indian-buddhism/stats.json` | `graph-topology-injection/ethics_packs/ethics-indian-buddhism/stats.json@9755f5e` | 186 | `f9580eb1704f5b28ba4d90848b5e6bd5f6ae87bae32ade0a98922f03ee4af685` |
| `ethics-indian-buddhism/triples.json` | `graph-topology-injection/ethics_packs/ethics-indian-buddhism/triples.json@9755f5e` | 8,835 | `774c5f48502cf1289a21d2e50ff132b6b2237a11d59b6429ae71fd4a92204c58` |
| `ethics-indian-buddhism/walk_encoding.txt` | `graph-topology-injection/ethics_packs/ethics-indian-buddhism/walk_encoding.txt@9755f5e` | 15,456 | `0bef713ed8167a6492d4c982f1812931048a64a7e696069abc647bcb17cb1973` |
| `feminism-aesthetics/stats.json` | `graph-topology-injection/ethics_packs/feminism-aesthetics/stats.json@9755f5e` | 182 | `c5225ba40816b5d2c12c8db253d0a7d86f30865d30e4ddd9f2581fe918112d7d` |
| `feminism-aesthetics/triples.json` | `graph-topology-injection/ethics_packs/feminism-aesthetics/triples.json@9755f5e` | 6,934 | `f5656360c6971357dab77260555d0dd009f72f10d1adaf86aa1f5186627a05e5` |
| `feminism-aesthetics/walk_encoding.txt` | `graph-topology-injection/ethics_packs/feminism-aesthetics/walk_encoding.txt@9755f5e` | 10,651 | `1d9de662685f53b72ad3977cd6f6ffbadddc9eb577c4219533364a4b9f1558f9` |
| `feminist-philosophy-biology/stats.json` | `graph-topology-injection/ethics_packs/feminist-philosophy-biology/stats.json@9755f5e` | 191 | `9c7e84611aedbad7b1dd820b2ed5047c21e36039bbb4a8bb30c836169fbde1ee` |
| `feminist-philosophy-biology/triples.json` | `graph-topology-injection/ethics_packs/feminist-philosophy-biology/triples.json@9755f5e` | 11,689 | `430c26b0a0d730635f83de8bf0514a515e93522de1a3c58f4d22d37c3221aa6f` |
| `feminist-philosophy-biology/walk_encoding.txt` | `graph-topology-injection/ethics_packs/feminist-philosophy-biology/walk_encoding.txt@9755f5e` | 23,958 | `00b9a486e23cff8635b697a1d6a8b182cd29006c866839c48c1cc61ae35ef331` |
| `informed-consent/stats.json` | `graph-topology-injection/ethics_packs/informed-consent/stats.json@9755f5e` | 180 | `21754bcc7a0aad3aa568115bafaeb9745638c7e5db577cc84cb41aed9a323b51` |
| `informed-consent/triples.json` | `graph-topology-injection/ethics_packs/informed-consent/triples.json@9755f5e` | 9,577 | `69348eac6efc354c1fc72dcc98c2fcd87eba6ad4946260d4c23ac74983f43f97` |
| `informed-consent/walk_encoding.txt` | `graph-topology-injection/ethics_packs/informed-consent/walk_encoding.txt@9755f5e` | 20,811 | `0909748ce43be1e2ba39af9a9a723eebbf5115ee7124f42c90d16ec97c90535c` |
| `justice-climate/stats.json` | `graph-topology-injection/ethics_packs/justice-climate/stats.json@9755f5e` | 178 | `93c0e287fb555f1f7f4ff5b6d45a86080d9a9e3e02758e694cc2b42a81775bd4` |
| `justice-climate/triples.json` | `graph-topology-injection/ethics_packs/justice-climate/triples.json@9755f5e` | 6,660 | `0735a38b31311f692cd719029d844e1002e4d36efd32b04692afa454d52acbc1` |
| `justice-climate/walk_encoding.txt` | `graph-topology-injection/ethics_packs/justice-climate/walk_encoding.txt@9755f5e` | 11,454 | `18166696ff995ca10e29bb376c8e23d2424101e3d4ec6749c576d6ec98217458` |
| `manifest.json` | `graph-topology-injection/ethics_packs/manifest.json@9755f5e` | 6,025 | `39771f81183c464b4e25da6bb12deb8384759120eaf8dc620fcac3a137b401bd` |
| `moral-animal/stats.json` | `graph-topology-injection/ethics_packs/moral-animal/stats.json@9755f5e` | 175 | `6b581bf439f72ddd294732ab55937875c20127ea64391f63f587d99f46919e60` |
| `moral-animal/triples.json` | `graph-topology-injection/ethics_packs/moral-animal/triples.json@9755f5e` | 6,546 | `72262fbc986fb9f31096092e8615687b41be4581fbbc6c0f62bfde3e852cc132` |
| `moral-animal/walk_encoding.txt` | `graph-topology-injection/ethics_packs/moral-animal/walk_encoding.txt@9755f5e` | 12,775 | `6597269a83173889d8b2242093fc397ea44d91b295ca42c2b6fe4c4de9a56bf1` |
| `moral-anti-realism/stats.json` | `graph-topology-injection/ethics_packs/moral-anti-realism/stats.json@9755f5e` | 182 | `d0fbd287ac8ec85343c5467dcd7df4055223de74aed3029b5e246c667d1e30ce` |
| `moral-anti-realism/triples.json` | `graph-topology-injection/ethics_packs/moral-anti-realism/triples.json@9755f5e` | 9,070 | `0ed61529e204507d01c53289feb5c6a5747ca26d36855ffc9617dd08c8e36f24` |
| `moral-anti-realism/walk_encoding.txt` | `graph-topology-injection/ethics_packs/moral-anti-realism/walk_encoding.txt@9755f5e` | 17,881 | `feb2f4f9b50d52237e1b0679fab8c3831f122090cfb3922b64a3f0c4a6cf123c` |
| `moral-character-empirical/stats.json` | `graph-topology-injection/ethics_packs/moral-character-empirical/stats.json@9755f5e` | 187 | `ae66172aad85dc5bc4dfd970672850d6446c1f1276bfb4fc4427eb6cc2e5cc34` |
| `moral-character-empirical/triples.json` | `graph-topology-injection/ethics_packs/moral-character-empirical/triples.json@9755f5e` | 6,324 | `77aa666c74b78cc902a1d52cdea9e84cf1714f209edb7c1bc88d17c58eb0385c` |
| `moral-character-empirical/walk_encoding.txt` | `graph-topology-injection/ethics_packs/moral-character-empirical/walk_encoding.txt@9755f5e` | 11,469 | `2265245e9e590b68fc846ddc2bf48704c20177fa1cf15de6b4b6f8cac76882a6` |
| `moral-character/stats.json` | `graph-topology-injection/ethics_packs/moral-character/stats.json@9755f5e` | 179 | `025c5b66e8616dc52ce5c8a9bf39da47fdee934b823170088c146ff8e37959d5` |
| `moral-character/triples.json` | `graph-topology-injection/ethics_packs/moral-character/triples.json@9755f5e` | 8,721 | `8344db80c55223512945a4dfcead76c967a3dd4f49189e17cd29bf63ef828191` |
| `moral-character/walk_encoding.txt` | `graph-topology-injection/ethics_packs/moral-character/walk_encoding.txt@9755f5e` | 17,863 | `3346b3ed4b4bdb20dc4cf26cb3f506c26f3318a8148ea3f851164bd954a24b74` |
| `moral-cognitivism/stats.json` | `graph-topology-injection/ethics_packs/moral-cognitivism/stats.json@9755f5e` | 181 | `4f5af507c79a2004143de2bf9dff784e79ac8ffaa254261987d93137a73bb272` |
| `moral-cognitivism/triples.json` | `graph-topology-injection/ethics_packs/moral-cognitivism/triples.json@9755f5e` | 9,041 | `75d802a1702c29cb83bb08eadc18ee360c2b577456067445464477478559b8a0` |
| `moral-cognitivism/walk_encoding.txt` | `graph-topology-injection/ethics_packs/moral-cognitivism/walk_encoding.txt@9755f5e` | 19,561 | `fe6fba67ef18c30cfbad0f128b8a92ccb7d6951f08a6295e309a14a0d9087bc3` |
| `personal-autonomy/stats.json` | `graph-topology-injection/ethics_packs/personal-autonomy/stats.json@9755f5e` | 181 | `3239689ee8a416cb0db3c2a9d93fc6c4d4b3fd022dffb770ae66f0bb83376d6e` |
| `personal-autonomy/triples.json` | `graph-topology-injection/ethics_packs/personal-autonomy/triples.json@9755f5e` | 8,871 | `6b98efa82f99616cc843d85b6a5873e2078b91c064bdd4d7b4af44e6f4bce6b9` |
| `personal-autonomy/walk_encoding.txt` | `graph-topology-injection/ethics_packs/personal-autonomy/walk_encoding.txt@9755f5e` | 20,694 | `013890493525641c6be5d8ca78376682dce09fa660126838a6952e9cc56b2df5` |
| `reasoning-moral/stats.json` | `graph-topology-injection/ethics_packs/reasoning-moral/stats.json@9755f5e` | 179 | `7a2e32ed3e2c0a294da01b3d51754d611f43f75782a168815663e84b8915aed7` |
| `reasoning-moral/triples.json` | `graph-topology-injection/ethics_packs/reasoning-moral/triples.json@9755f5e` | 8,073 | `9919bec3ab02e44104fb2e2980959af304f5ea099bc35ec311c3b338060ea709` |
| `reasoning-moral/walk_encoding.txt` | `graph-topology-injection/ethics_packs/reasoning-moral/walk_encoding.txt@9755f5e` | 18,620 | `0977e605999558d6f18cb5a290e00e901f9ba85016730c2d43367e0de6625e54` |
