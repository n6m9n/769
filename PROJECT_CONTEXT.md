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

## 5. Repository map and status (final phase, 2026-11-21)

```
motok/pretokenize.py        regex pretokenizer, (parallel) counting
motok/toast/trees.py        exact level-wise n-gram counts, split trees, inference, tokenizer
motok/toast/problem.py      head trees per language + SAA tail samples (tail_samples), morph marks
motok/lp/model.py           VocabLP: LP build (vectorised), scalarisation, warm/cold solve, bases, rounding, evaluation;
                            tail_tokens='head' blocks candidates that only occur in sampled tail trees
motok/baselines/            hf.py (BPE/Unigram x alpha), parity_bpe.py (authors' learner in ext/), morph_bpe.py (reimpl.)
motok/eval/metrics.py       compression, FLORES tokens/sentence + premiums, Gini, Renyi, morph P/R/F1, hypervolume
motok/morph/segment.py      Morfessor; SIGMORPHON + MorphScore v2 gold loaders
motok/tokload.py            one loader for every tokenizer file
scripts/01..08              download, prepare, baselines, sweep (--phase anchor|sweep|merge), evaluate, LM, report
scripts/run_lp.sh           parallel LP pipeline; run_lm_queue.sh (GPU queue); link_lp.sh (link LP tokenizers)
report/REPORT.md            final report (+ figures/, tables.md)
```

Server runs live in `/data1/samarpitb/motok/results/` (not in git): `full` (baselines, eval.json, lm/),
`full_10k` (head-only), `full_5k_t5k`, `full_10k_t10k` (SAA, main), `full_20k(_t20k)`, `full_40k`, `full_10k_morph`,
`*_ngram` (biased SAA rule, kept as evidence).

## 6. Final results (see report/REPORT.md)

- Near-integral at scale: 123 points in the three main runs, |gap| <= 3e-4, 0 bound violations (bound shading).
- Head-only fronts (10k/lang) are large on the training side (-21 % worst language) but do not transfer: held-out
  the LP loses to BPE (FLORES mean 44.7 vs 39.5) and the min-max tokenizer is the least fair.
- SAA of the tail + head-candidate rule: LP estimate within 1-4 % of the exact cost (naive SAA: 24 % too low for ru);
  held-out hypervolume 0.08 -> 0.75 (BPE alpha-sweep 0.78).
- SAA 10k+10k parity front transfers: worst-language held-out cost -13 % (FineWeb) / -14 % (FLORES) at flat mean;
  min-max Gini 0.004 (lowest of all tokenizers). Parity-aware BPE still fairest in absolute terms.
- LMs (35 M params, 3 seeds): MorphBPE 1.110 bpb best; BPE alpha=0 worst baseline (1.153); SAA LP tokenizers worse
  (1.19) than head-only LP (1.139) despite better compression.
- Morph-first trees: 0 spurious boundaries on the training side for +2.2 % Ttot, but held-out compression collapses.

## 7. Known issues / open questions

1. Candidate set is the bottleneck: split-tree nodes only. Next: external candidates (BPE merges, Unigram pieces).
2. SAA LM penalty (+4.5 % bpb vs head-only LP) is unexplained; hypothesis: word-dependent fragmentation.
3. Min-max anchors are slow (5-7x a compression solve; 2.4 h at 20k head-only); warm primal start inconclusive.
4. SAA uses one sample; no independent re-estimation / statistical SAA gap yet.
5. Matched-bytes LM protocol gives more steps to tokenizers with more tokens (MorphBPE +7 %, Unigram +17 %).
6. Parity-aware BPE is tuned on FLORES+ dev (in-distribution for devtest); its two variants give identical vocabularies.
7. Indic gold morphology is small (MorphScore mr: 178 words); MorphBPE is a reimplementation.

## 8. Roadmap status

Done: P0 (smoke, FLORES, full data), P1 (Parity-aware BPE, MorphBPE, hypervolume, ablation, scaling, weighted sums),
P2 (LM eval). Not done: sharing objective study, script-budget constraints, cutting planes (gaps never needed them).

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
