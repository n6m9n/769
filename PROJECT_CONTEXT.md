# Project context: Multi-objective optimisation for multilingual tokenization

This file is the single source of truth for the project. Read it fully before changing code.
`CLAUDE.md` holds the short operating rules; this file holds the why, the math, the status and the plan.

---

## 1. Course and deliverables

- Course: **Optimization in Machine Learning** (CS 769). Team project.
- Official topic: formulate tokenizer learning as a multi-objective optimisation problem that jointly
  optimises compression, morphological alignment, cross-lingual vocabulary sharing and downstream
  performance, and show it balances these better than frequency-based methods like BPE.
- **Interim presentation: week of Oct 12–16, 2026.** 5 minutes (4 talk + 1 Q&A), max 5 slides:
  1. Problem & Motivation, 2. Related Work (2–3 papers, how we differ), 3. Dataset & Approach,
  4. Progress So Far (incl. preliminary results), 5. Next Steps & Challenges.
- Final submission date: not yet known. Assume ~5 weeks after the interim.

Immediate goal: by ~Oct 11, have **real figures for Slide 4** (Pareto fronts vs baselines, per-language
cost bars, LP timings / integrality gaps) from the smoke run and, if possible, a first full-data run.

---

## 2. Research framing

Frequency-based tokenizers (BPE, Unigram) optimise one thing (compression) greedily, and the dominant
language in the training mix wins: low-resource / non-Latin languages pay a token premium, and subword
splits ignore morphology. We make the competing goals **explicit objectives of one optimisation problem**.

Research questions:
- **RQ1** Can compression, cross-lingual parity and morphological alignment be optimised jointly in one LP,
  and does the LP relaxation stay near-integral when objectives couple languages?
- **RQ2** What do the Pareto fronts and shadow prices (exchange rates between objectives) look like across
  typologically different languages?
- **RQ3** Does the LP front dominate what frequency-based methods reach (BPE/Unigram with temperature
  sampling, Parity-aware BPE, MorphBPE)? Compare with the hypervolume indicator.
- **RQ4** Which Pareto points give the best downstream bits-per-byte in small LMs?

---

## 3. Method: "Pareto-ToaST"

Builds on ToaST (Tokenization with Split Trees, Schmidt et al. 2026, arXiv 2605.22705):
- Each frequent pretoken gets a fixed full binary **split tree** built only from n-gram counts
  (split position maximises min(count(left), count(right))). Leaves are single characters.
- Inference: descend the tree; emit the first in-vocabulary node on each root-to-leaf path.
- Vocabulary selection is an IP whose LP relaxation is almost always integral.

### Variables and core constraints
- `x_i ∈ [0,1]`: candidate token i in the vocabulary (base characters fixed to 1).
- `z_k ∈ [0,1]`: node k (of tree j) is emitted.
- `Σ_i x_i = m − 256` (256 byte-fallback tokens are reserved outside the LP).
- For every leaf: `Σ_{k on leaf→root path} z_k = 1`.
- `z_k ≤ x_{tok(k)}`.

### Aggregate columns (the multi-objective layer)
Each objective is a column defined by an equality row, all linear in z or x:

| column | meaning | units |
|---|---|---|
| `T_l` | tokens of language l = 1000·κ_l/chars_l · Σ_k c_{j(k),l} z_k | tokens per 1000 English-equivalent chars |
| `M_l` | spurious boundaries: same sum restricted to nodes whose start is strictly inside a morpheme | per 1000 chars |
| `A_l` | correct interior morpheme boundaries hit | per 1000 chars |
| `Ttot`, `Mtot` | Σ_l π_l T_l, Σ_l π_l M_l, π_l = data share of chars | |
| `t` | worst-language cost, via rows `T_l − t ≤ 0` | |
| `S` | sharing score Σ_i s_i x_i, s_i = (#languages whose trees contain token i) − 1 | |

κ_l = chars of FLORES text in l / chars of the same content in English (content normalisation; 1.0 until
FLORES is downloaded). c_{j,l} = count of pretoken j in language l.

**Why aggregate columns:** any scalarisation only changes costs or bounds on a handful of columns:
- weighted sum: costs on `T_l`, `M_l`, ...
- ε-constraint: upper bounds on `Mtot`, `t` (= T_l ≤ τ for all l)
- min-max parity: cost on `t`

So sweeps warm-start (bound changes keep the old basis dual-feasible → dual simplex), and the **reduced
cost of a bounded aggregate column is d(Ttot)/d(bound)**, the local exchange rate between objectives.
For an LP, weighted sums recover the entire efficient set of the relaxation.

### Morphology objective, precisely
A node is "misaligned" if its start position is strictly inside a morpheme (a spurious token boundary).
Counting only left edges avoids double counting. Words emitted whole cost 0. Therefore cost 0 ⇔ every token
is a concatenation of whole morphemes. Compression and morphology only conflict in the subword regime,
which is where low-resource and agglutinative languages spend most of their tokens.
(MorphBPE instead forbids tokens crossing morpheme boundaries; different notion, keep that in mind when comparing.)

### Rounding and certificates
ToaST rounding: keep x ≈ 1, fill remaining slots with fractional tokens of largest objective contribution.
Node weights use |row duals| of the aggregate definitions, so the rule works for any scalarisation.
`lp.evaluate(vocab)` runs real inference and recomputes every aggregate exactly. LP value = lower bound;
rounded value = upper bound; the gap is a per-point optimality certificate.

### Tree construction hierarchy
Split positions are chosen from the highest non-empty level of a preference hierarchy:
`morph > grapheme > char` (ToaST Appendix A). Default config is `[grapheme, char]` (count-based trees).
The soft morph penalty can only choose among splits the tree offers, so tree construction is an
ablation axis: (A) count-based trees + penalty vs (B) morph-first trees + penalty. (B) needs a segmenter
at inference for unseen words (latency cost, as MUTANT reported).

---

## 4. Data and languages

| code | language | script | why |
|---|---|---|---|
| en | English | Latn | high-resource, simple morphology |
| hi | Hindi | Deva | shares script with Marathi (sharing pair) |
| mr | Marathi | Deva | more agglutinative, lower-resource |
| ta | Tamil | Taml | Dravidian, highly agglutinative, own script |
| ru | Russian | Cyrl | fusional |
| hu | Hungarian | Latn | agglutinative; shares script with English |

- **Training text:** FineWeb (English, `sample-10BT`) and FineWeb-2 (others). Download sizes in
  `configs/languages.yaml` are deliberately imbalanced (200/100/75/50/50/25 MB) to create a fairness problem.
- **Parallel eval:** FLORES+ dev/devtest (`openlanguagedata/flores_plus`, **gated**: accept terms, `huggingface-cli login`).
- **Gold morphology (evaluation only):** SIGMORPHON 2022 word-level for en/ru/hu (downloaded from GitHub).
  Indic gold: check MorphScore coverage (github.com/catherinearnett/morphscore, 70+ languages from UD).
- **Morphology training signal:** unsupervised Morfessor Baseline per language (type-based).
  Never evaluate morphology against Morfessor output; that would be circular.
- **Smoke data:** UD treebank sentences (GitHub) for all six languages. Small: mr and ta have only ~350
  training sentences. Good for testing code, not for claims.
- Pretokenizer: ToaST's length-limited GPT-4o regex (letter classes include `\p{M}`, so Indic vowel signs
  stay attached). Trees are character-level: never split inside a UTF-8 character.

---

## 5. Repository map and status

```
motok/pretokenize.py     regex pretokenizer, counting                                     tested
motok/toast/trees.py     n-grams, split trees, hierarchy, inference, SplitTreeTokenizer   tested
motok/toast/problem.py   per-language tree selection, balanced n-grams, morph marks       tested (smoke)
motok/toast/io.py        save/load trained split-tree tokenizers                          used by sweep
motok/lp/model.py        VocabLP: LP build (HiGHS), scalarisation, warm/cold solve,
                         rounding, exact evaluation                                       tested
motok/morph/segment.py   Morfessor training/segmenter, SIGMORPHON gold loader             smoke-tested
motok/baselines/hf.py    HF byte-level BPE / Unigram, same pretokenizer, temperature mix  smoke-tested
motok/eval/metrics.py    compression, Gini, Rényi eff., morph P/R/F1 + MorphScore,
                         sharing (Jaccard), nondominated filter, hypervolume              unit-tested (Pareto utils only)
scripts/01_download.py   smoke | morph | fineweb | flores                                smoke + morph run; fineweb/flores UNTESTED
scripts/02_prepare.py    counts, Morfessor, problem.pkl                                  run on smoke
scripts/03_baselines.py  BPE/Unigram × alpha                                             run on smoke
scripts/04_sweep.py      anchors + parity / morph / grid ε-sweeps                        all run on smoke (2026-10-06); infeasible points recorded, not rounded
scripts/05_evaluate.py   held-out metrics for every tokenizer                            run on smoke
scripts/06_plots.py      fronts, per-language bars, shadow prices, summary.md            run on smoke
scripts/check_results.py LP<=integer, monotone fronts, full round trip -> checks.json      run on smoke
tests/test_core.py       10 tests incl. brute-force LP optimality on a toy                passing
```

---

## 6. Preliminary results (smoke run, UD text)

Setup: 6 languages, 2000 trees per language → 10,704 trees, 144,608 nodes, 288 base chars, vocab 4000.
Pretoken coverage of the selected trees: en 79%, hi 83%, mr 100%, ta 88%, **ru 64%, hu 72%**
(agglutinative/fusional languages need more trees for the same coverage, a known ToaST limitation).

English only (3000 trees): LP integral at m = 1000 and 2000 (inference matches LP to 1e-13), gap 1.5e-4 at m = 1500.

Multilingual (training-side LP values, tokens per 1000 chars, κ = 1):

| point | Ttot | worst lang (Tmax) | Mtot | gap | time |
|---|---|---|---|---|---|
| compression anchor (= multilingual ToaST) | 300.5 | 633.3 | 44.0 | 0 (integral) | 13.6 s cold |
| parity_4 (τ ≈ 437) | 308.8 (+2.7%) | 437.5 (−31%) | 49.7 | 2.1e-4 | 16.5 s warm |
| parity_5 (τ ≈ 398) | 322.5 (+7.3%) | 398.5 (−37%) | 58.9 | 5.0e-5 | 21.9 s warm |
| min-max anchor | 358.9 | 359.1 | 84.7 | 1.2e-5 | 75.2 s cold |
| min-morph anchor | 394.3 | 640.8 | 32.1 | 0 | 13.0 s cold |
| morph_0 | 300.7 | 631.5 | 42.3 | 1.0e-4 | 10.3 s warm |

Takeaways so far:
- Near-integrality survives cross-language coupling constraints (all gaps ≤ 4e-4). (RQ1)
- The fairness–compression front is very favourable at first: −31% worst-language cost for +2.7% total. (RQ2)
- Warm-started sweep points: 0.5–22 s vs 13–75 s cold.

Caveats: κ = 1 (no FLORES yet), so cross-script comparison isn't content-normalised; tiny mr/ta data;
these are training-side numbers.

Full smoke pipeline incl. held-out eval, baselines, checks and diagnosis (2026-10-06): κ = 1 run in
`results/smoke_kappa1/REPORT.md` (numbers above reproduce to ±0.1); **FLORES κ run** in `results/smoke/REPORT.md`
(not in git). FLORES+ downloaded and verified aligned; κ: en 1.000, hi 1.001, mr 1.004, ta 1.167, ru 1.103, hu 1.055.
With κ: parity_4 −29% worst-language for +4.2% total; FLORES devtest: parity cuts worst language −15% and mean −9%
vs the LP compression anchor, but BPE α=0 still wins in absolute tokens (coverage, issue 8).

---

## 7. Known issues

1. **Morph ε-sweep stall:** warm dual simplex hung (> 5 min) on `morph_1` (older highspy). With highspy 1.15.1
   it does not reproduce: all morph points solve warm in 4–7 s (2026-10-06). The 60 s warm-start limit with cold
   fallback stays as a safety net; HiGHS `time_limit` is *cumulative* over `run()` calls, so `solve()` offsets it by
   `getRunTime()` (before that fix every warm start fell back to cold instantly). Fallbacks are logged per point.
2. **Warm-start rule:** after objective (cost) changes use `solve(warm=False)` (presolve) or
   `method="primal"`; after bound/RHS changes use `warm=True, method="dual"`. Sweep bounds loose → tight.
   A bound change before the first solve makes HiGHS think it has a basis and skip presolve; `solve()`
   clears the solver on the first call to avoid this.
3. **Scaling bottlenecks:** leaf-coverage rows are built in a Python loop (vectorise before >100k trees);
   `ngram_counts` keeps all substrings in memory (chunk/prune for full FineWeb); `build_problem` merges all
   pretoken types in a Python dict.
4. `encode_with_offsets` gives byte-fallback pieces the span of their character (fine for boundary metrics).
5. `_ScriptRouter` in `toast/io.py` (morph-first inference) just tries segmenters in order: OK for
   script-disjoint languages, approximate for hi/mr.
6. HF Unigram training is slow (~10–45 s each on smoke data); expect much longer on full data.
7. **Rounding overshoots ε-bounds** slightly (smoke: 17/22 points, Tmax ≤ +0.45, Mtot ≤ +0.12 per 1000 chars), so
   `gap_Ttot` is not a certificate at those points. Needs bound-aware rounding / repair.
8. **Coverage dominates held-out results:** with 2000 trees/lang (coverage 64–100%) the LP beats BPE by 9% on the
   selected pretokens but loses on held-out text (428 vs 380). Trees must cover most of the text, and the
   vocab must stay well below the number of trees (else every tree root fits in the vocabulary and the front
   collapses).
9. **Gold morphology barely overlaps the trees** (smoke: < 1% of SIGMORPHON test words are trees), so held-out
   morph metrics can't reflect the morph objective yet.
10. Grid sweep uses a global ε range; tight-τ corners are infeasible (min Mtot | Tmax ≤ 427.5 = 40.8 on smoke).
    Choose ε per τ from [min Mtot | τ, Mtot at τ]. Grid also visits τ tight → loose (against rule 2).

---

## 8. Roadmap (priority order)

**P0, before the interim (target Oct 11):**
1. Run the smoke pipeline 01→06 end to end; confirm the morph sweep finishes; produce `summary.md` and figures.
2. Sanity-check results: round-trip failures = 0, LP ≤ integer for every point, fronts look monotone.
3. Download FLORES+ (κ and premiums) and a first FineWeb slice; run `configs/full.yaml` at
   **5k trees/language**, vocab 32k. Record LP build/solve time and memory. Scale only if it fits.
4. Produce Slide-4 figures: parity front vs BPE/Unigram, per-language bars, timing/integrality table.

**P1, after the interim:**
5. Add **Parity-aware BPE** (github.com/swiss-ai/parity-aware-bpe) and **MorphBPE**
   (github.com/qcri/MorphBPE) baselines with the same data and vocab size; hypervolume comparison.
6. Tree-construction ablation: `[grapheme,char]` vs `[morph,grapheme,char]`.
7. Scaling work (vectorise LP build, chunked n-grams) → 20k–40k trees/language.
8. Weighted-sum sweeps (warm primal) as a second way to trace the front; compare with ε-constraint.

**P2:**
9. Small-LM downstream eval: ~25M-param GPT, same vocab size and **same byte budget** for every tokenizer;
   per-language bits-per-byte on held-out text; 4–6 tokenizers (Pareto points + baselines).
10. Sharing objective and script-budget constraints (`Σ_{i∈script} x_i ≥ b_s`).
11. If rounding gaps grow at scale: repair heuristic / cutting planes (see Nichol's blog, "cycle constraints").

---

## 9. Working rules

- Keep `pytest -q` green. Add a test for every new objective/constraint (at minimum: integral LP ⇒
  `evaluate(round)` equals LP aggregates; LP ≤ integer).
- Don't change units or column semantics without updating this file.
- **Ask before:** downloads > 1 GB, any run expected to take > 30 min, adding heavy dependencies,
  deleting results.
- Long runs: run in the background with logs to `results/<name>/*.log`; report progress, don't block.
- Record every experiment's config, timings, gaps and git commit in its results folder.
- Report results honestly with caveats (training-side vs held-out, κ status, data size).
- Prefer small, reviewable commits with clear messages.

---

## 10. References

- ToaST: Schmidt et al. 2026, *Tokenization with Split Trees*, arXiv 2605.22705 (core formulation; App. A
  hierarchy + multilingual reweighting; App. B.5 warm starts).
- ConvexTok: Tempus et al. 2026, *Tokenisation via Convex Relaxations*, arXiv 2605.22821 (LP over all
  tokenizations; certified within ~1% of optimal).
- Nichol 2026, *Finding Optimal Tokenizers* (blog.aqnichol.com): cutting planes for the tokenization ILP.
- Parity-aware BPE: Foroutan et al., ACL 2026 (arXiv 2508.04796): greedy fair-max merges; Gini of per-language costs.
- MorphBPE: Asgari et al., Findings ACL 2026: merges forbidden across morpheme boundaries; morph edit
  distance and consistency F1.
- MUTANT: Rana et al., ACL 2026: multilingual (Indic) tokenizer recipe, script-wise vocab allocation,
  fertility / NSL / Rényi / bytes-per-token evaluation.
- MorphScore: Arnett & Bergen 2025; Arnett et al. 2025 (70 languages): morph alignment explains little
  downstream variance, so treat morphology as a trade-off to expose, not a quantity assumed to help.
