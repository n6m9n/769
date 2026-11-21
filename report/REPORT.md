# Multi-objective vocabulary selection for multilingual tokenization

CS 769 (Optimization in Machine Learning), final report. Code: https://github.com/n6m9n/769

## Abstract

We cast multilingual tokenizer vocabulary selection as a multi-objective linear program (LP) over split trees
(ToaST, Schmidt et al. 2026), with compression, worst-language cost (cross-lingual parity), spurious morpheme
boundaries and cross-lingual sharing as *aggregate columns*. Every scalarisation (weighted sum, ε-constraint,
min-max) only changes costs or bounds of a handful of columns, so whole Pareto fronts are traced with
warm-started simplex, reduced costs give exchange rates between objectives, and LP-vs-rounded values give a
per-point optimality certificate. On 500 MB of FineWeb / FineWeb-2 text in six typologically diverse languages
(en, hi, mr, ta, ru, hu) we solve LPs with up to 3.4 M node variables, compare against BPE / Unigram with
temperature sampling, Parity-aware BPE and MorphBPE, and train 35 M-parameter LMs per tokenizer.

Main findings. (1) The LP stays (near-)integral at scale: across >100 Pareto points all integrality gaps are
≤ 3·10⁻⁴, most points are exactly integral, and with *bound shading* every rounded vocabulary satisfies its
ε-bounds, so every gap is a certificate. (2) Fronts computed on the head of the distribution (the standard ToaST
problem) look very favourable on the training side (−21 % worst-language cost for +8 % total tokens) but **do not
transfer to held-out text**, where the LP loses to BPE. (3) A **sample average approximation (SAA)** of the
pretoken tail fixes the objective, but naive SAA suffers a strong optimizer's curse (the LP's own estimate is 24 %
too optimistic for Russian); restricting tail trees to head-tree candidate tokens removes it (estimate within
1–4 % of the exact cost) and improves held-out compression by 8 % at half the problem size. (4) With a
correctly specified objective, the remaining compression–fairness–morphology conflicts *within the split-tree
candidate set* are small; the remaining gap to BPE comes from candidate generation and fixed trees, not from the
optimisation. (5) In downstream LMs, differences are ~1–4 % bits per byte, the best LM uses MorphBPE, and the
most compressive tokenizer (BPE, uniform sampling) gives the worst LM.

---

## 1. Problem and formulation

Frequency-based tokenizers (BPE, Unigram) optimise one thing (compression) greedily; the dominant language of the
training mix wins, non-Latin and low-resource languages pay a token premium, and splits ignore morphology. We make
the competing goals explicit objectives of one optimisation problem.

**Split trees (ToaST).** Every frequent pretoken *j* gets a fixed full binary tree whose leaves are characters;
each internal node is split where min(count(left), count(right)) is largest (counts = language-balanced n-gram
counts). Inference: walk down the tree and emit the first in-vocabulary node on each root-to-leaf path.

**LP.** Variables x_i ∈ [0,1] (token i in the vocabulary; base characters fixed to 1) and z_k ∈ [0,1] (node k is
emitted). Core constraints: Σ_i x_i = m − 256 (256 byte-fallback tokens are reserved), Σ_{k on leaf→root path} z_k = 1
for every leaf, z_k ≤ x_{tok(k)}.

**Aggregate columns** (all linear in z or x, each defined by one equality row):

| column | meaning |
|---|---|
| T_l | tokens per 1000 English-equivalent chars of language l: 1000·κ_l/chars_l · Σ_k c_{j(k),l} z_k |
| M_l | spurious morpheme boundaries (nodes that start strictly inside a morpheme), same units |
| A_l | correctly hit morpheme boundaries |
| Ttot, Mtot | data-share weighted sums of T_l, M_l |
| t | worst-language cost, rows T_l − t ≤ 0 |
| S | sharing score Σ_i s_i x_i |

κ_l is the content normalisation from FLORES+ dev (chars of l per English char on the same sentences: en 1.000,
hi 1.001, mr 1.004, ta 1.167, ru 1.103, hu 1.055). Morphology comes from unsupervised Morfessor models (training
signal only); evaluation uses gold MorphScore v2 (UD, all six languages) and SIGMORPHON 2022 (en/ru/hu).

**Scalarisations are cost/bound changes on aggregate columns:** weighted sum = costs on Ttot, t, Mtot;
ε-constraint = upper bounds on t (parity) or Mtot (morphology); min-max = cost on t. Bound changes keep the
previous basis dual feasible (dual simplex warm start), cost changes keep it primal feasible (primal simplex),
and the reduced cost of a bounded aggregate column is dTtot/d(bound), the local exchange rate.

**Rounding and certificates.** Keep x ≈ 1, fill remaining slots by LP contribution (weights from the duals of
the aggregate rows), then run *exact* split-tree inference to recompute all aggregates. LP value = lower bound,
rounded value = upper bound.

## 2. What we added for the final phase

1. **Bound shading** (new). Rounding could overshoot ε-bounds (smoke run: 14/20 points), so the gap was not a
   certificate there. If the rounded vocabulary violates its bound, we tighten the bound by the overshoot, re-solve
   warm and re-round (≤ 6 times). The LP value at the original bound stays the lower bound. Result: bound
   violations 0 / all points in every run.
2. **Parallel warm-started pipeline.** Three anchors solve cold in parallel and save HiGHS bases; every sweep
   (parity, morph, grid, two weighted-sum sweeps) runs in its own process warm from the compression basis.
   The 3-objective grid now chooses ε per τ in [min Mtot | τ, Mtot at min Ttot | τ] (no infeasible corners).
3. **Scaling.** Exact level-wise (Apriori) n-gram counting, vectorised LP construction (2.5 M rows in 13 s),
   parallel counting / Morfessor training.
4. **Sample average approximation of the tail** (new, §5).
5. **Baselines.** Parity-aware BPE (authors' learner, our pretokenizer, FLORES+ dev as parity set) and MorphBPE
   (reimplemented: no public code; merges never cross a Morfessor boundary).
6. **Gold morphology for all six languages** (MorphScore v2) and **downstream LMs** (§7).

## 3. Setup

- **Training text:** FineWeb sample-10BT (en) and FineWeb-2 (others), deliberately imbalanced:
  en 200 MB, hi 100, ru 75, hu 50, mr 50, ta 25 (393 M characters). Held-out: first 2000 documents per language.
- **Parallel evaluation:** FLORES+ devtest (1012 sentences; same content in every language).
- **Pretokenizer:** ToaST's length-limited GPT-4o regex (keeps Indic vowel signs attached), shared by all tokenizers.
- **Vocabulary:** 32 000 for every tokenizer (LP: 31 744 + 256 byte fallback).
- **Compute:** 96-core / 1 TB server, 3× RTX A6000; HiGHS 1.16 (dual / primal simplex).

## 4. Head-only LP: favourable fronts that do not transfer

[fig_train_fronts_lp10k.png] — training-side fronts at 10k trees per language (54 k trees, 769 k nodes).

At 10k trees/language the parity front is very favourable on the training side: worst-language cost 324 → 271
(−16 %) for +0.4 % total tokens, and 256 (−21 %) at the min-max anchor (+8.3 %). Every point is near-integral
(|gap| ≤ 3.4·10⁻⁵ on the parity and morph sweeps), weighted-sum and ε-constraint points lie on the same front, and
warm points take 20–200 s against 300–1700 s for cold anchors.

**Held-out, the LP loses to BPE** (FLORES tokens per sentence, mean / worst language):

| tokenizer | FineWeb T mean | FLORES mean | FLORES worst |
|---|---|---|---|
| BPE α = 1 | 308.9 | 39.5 | 48.4 |
| BPE α = 0 | 296.8 | 37.8 | 41.2 |
| LP 10k, compression anchor | 348.8 | 44.7 | 51.7 |
| LP 10k, best parity point | 343.3 | 44.0 | 49.2 |
| LP 10k, min-max anchor | 354.3 | 45.5 | 61.0 |

The training-side parity gain mostly vanishes (worst language 406.6 → 403.5 on FineWeb) and the min-max anchor is
*worse* held-out. Cause: trees cover only 66–89 % of the pretoken mass (ru/hu 72 %); the LP never sees the long
tail, spends the vocabulary on whole frequent words, and the fairness gains are bought with whole words of the
worst language that do not help unseen text. More trees help coverage only slowly (5k → 40k trees per language:
mean coverage 71 % → 88 %, worst 59 % → 80 %) while solve time grows ~ n^1.4 (§8).

## 5. Sample average approximation of the tail and its optimizer's curse

We add, per language, S trees for tail pretokens drawn with probability ∝ count (with replacement), each draw
weighted by (tail mass)/S. Then Σ_j c_{j,l} · tokens_j is an unbiased estimate of the token count over *all* text
for any *fixed* vocabulary.

**Optimizer's curse.** The vocabulary is not fixed: the LP chooses it using the same sample. With any frequent
n-gram allowed as a token, each sampled tail word carries the weight of a top-2000 word, and the LP buys tokens
that cover individual sampled words. Measured on 5k head + 5k tail per language (LP estimate / exact cost over the
full training distribution, κ-normalised):

| candidate rule | en | hi | mr | ta | ru | hu |
|---|---|---|---|---|---|---|
| any frequent n-gram | 232 / 269 | 290 / 302 | 293 / 312 | 379 / 396 | **290 / 381** | **289 / 353** |
| head-tree tokens only | 272 / 277 | 284 / 288 | 282 / 289 | 341 / 351 | 357 / 369 | 332 / 340 |

Restricting tail trees to candidate tokens that are nodes of head trees (tail trees reweight the objective, they
never add candidates) removes the bias (estimate within 1–4 %), lowers the *exact* cost (ta 396 → 351, hu 353 → 340)
and makes the LP easier (55 % of token variables fixed to 0; compression anchor 75 s instead of 225 s). The exact
tokenizer and the LP agree node for node (0 mismatches on 20 000 trees), and the exact training cost matches the
held-out cost (ru: 381 vs 380), so the remaining error is purely the SAA estimate.

**Held-out effect** (5k head + 5k tail, half the trees of the 10k head-only problem):

| tokenizer | FineWeb T mean | FLORES mean | FLORES worst |
|---|---|---|---|
| LP 10k head-only | 348.8 | 44.7 | 51.7 |
| LP 5k+5k SAA, n-gram rule | 335.8 | 43.0 | 51.2 |
| **LP 5k+5k SAA, head rule** | **319.5** | **41.0** | **45.6** |
| LP 10k+10k SAA, head rule | 318.6 | 40.8 | 49.0 |
| LP 20k+20k SAA, head rule | 319.1 | 40.9 | 49.9 |

SAA cuts held-out tokens by 8 % at half the problem size; the LP's worst language on FLORES (45.6) now beats BPE
α = 1 (48.4). Scaling the head beyond 10k does not help further.

## 6. Fronts with a correctly specified objective

[fig_train_fronts.png] — training side, SAA 5k+5k (head rule).

With the objective covering all text, the fronts within the split-tree candidate set become small: on 5k+5k the
min-max anchor lowers the worst-language cost only from 356.5 to 354.6 (+0.01 % Ttot) and the morph front spans
62.9 → 61.6 spurious boundaries per 1000 chars (+2.1 % Ttot). All 35 points are integral or nearly so
(|gap| ≤ 2.5·10⁻⁴, 0 bound violations), and both scalarisations agree. *TBD: 10k+10k fronts.*

Interpretation: the large head-only trade-offs were largely an artefact of the mis-specified objective (they
traded whole head words of one language for another). Once the objective counts all text, the binding limit is the
candidate set: the worst language (Tamil, κ = 1.167) can only get cheaper if it has more useful candidate subwords
in its trees, which the LP cannot create. This points to *candidate generation* (not the optimiser) as the place to
improve — e.g. adding BPE-merge candidates or language-specific tree budgets.

## 7. Comparison with baselines

[fig_parity_front.png] [fig_morph_front.png] [fig_per_lang.png]

- **Compression:** BPE wins (data-weighted 276–282 vs LP SAA 302). Uniform temperature sampling (α = 0) moves BPE
  along a compression–fairness front (worst-language FLORES 48.4 → 41.2).
- **Parity:** Parity-aware BPE is the clear winner (FLORES worst 38.4 at mean 38.1; Gini 0.018). Caveat: it is tuned
  on FLORES+ dev, i.e. in-distribution for devtest; on FineWeb it is still the fairest (worst 312.1).
  Its base and moving-window variants learn the merges in a different order but end with the *identical*
  32k vocabulary (every test line tokenizes identically).
- **Morphology:** MorphBPE (MorphScore F1 0.536) and Unigram (0.50) align best; BPE ≈ 0.27–0.29; LP ≤ 0.29. The LP's
  morph objective moves the training-side spurious-boundary count but its held-out effect on gold morphology is
  small (F1 0.19 → 0.23 along the SAA morph sweep). Morph-first trees (ablation, §9) remove the training-side
  conflict but hurt held-out compression badly.
- **Hypervolume** (normalised; FLORES mean × worst): all baselines 0.99, BPE family 0.75, LP SAA 5k+5k 0.21.

## 8. Optimisation at scale

[fig_scaling.png]

| trees/lang | trees | nodes | coverage mean / worst | compression anchor | min-max anchor | warm point (median) |
|---|---|---|---|---|---|---|
| 5k | 27 k | 0.36 M | 71 % / 59 % | 65 s | – | – |
| 10k | 54 k | 0.77 M | 77 % / 66 % | 312 s | 1690 s | 101 s |
| 20k | 109 k | 1.6 M | 83 % / 73 % | 1171 s | TBD | – |
| 40k | 216 k | 3.4 M | 88 % / 80 % | 3323 s | – | – |

- All compression anchors are exactly integral; interior-point (IPX, HiPO) and PDLP were not faster than dual
  simplex on these LPs (5k: dual 50 s, parallel dual 42 s, HiPO 61 s, IPX 73 s, PDLP 87 s).
- Warm starts: sweep points cost 2–30 % of a cold solve; after a simultaneous cost *and* bound change the warm dual
  simplex stalls, so the grid first returns to the min-Ttot basis with a primal step.
- Min-max anchors are the bottleneck (5–7× a compression solve); warm-starting them from the compression basis
  with primal simplex: *TBD*.

## 9. Ablation: morph-first trees

With split positions preferring Morfessor boundaries ([morph, grapheme, char], 10k trees/language), the
compression anchor is *better* on the training side (Ttot 235.8 vs 236.7) with 4× fewer spurious boundaries
(2.0 vs 7.8), and the min-morph anchor reaches **0 spurious boundaries for +2.2 % Ttot**. Held-out, however, the
tokenizer needs Morfessor at inference for unseen words, and compression collapses (FineWeb T mean 423 vs 349,
FLORES worst 69 vs 52) while gold morphology improves only modestly (MorphScore F1 0.29 vs 0.24). This matches
MUTANT's observation that morphology-first segmentation is expensive at inference.

## 10. Downstream language models

[fig_lm.png]

35.5 M-parameter GPT (6 layers, d = 512, context 512, tied embeddings), the same 144 MB of text for every tokenizer
(24 MB per language, 4 epochs), same optimiser and tokens per step; bits per byte on held-out FineWeb and FLORES+;
3 seeds.

| tokenizer | FineWeb bpb (± sd) | FLORES bpb | worst-language bits / FLORES sentence | steps |
|---|---|---|---|---|
| MorphBPE | **1.110 ± 0.002** | **1.078** | 273.8 | 3283 |
| Unigram α = 1 | 1.120 ± 0.007 | 1.086 | **273.3** | 3572 |
| LP 10k head-only, morph point | 1.136 ± 0.002 | 1.110 | 277.7 | 3674 |
| LP 10k head-only, compression | 1.138 ± 0.002 | 1.111 | 278.9 | 3539 |
| Parity-aware BPE | 1.140 ± 0.010 | 1.104 | 277.7 | 3106 |
| BPE α = 1 | 1.143 ± 0.012 | 1.104 | 278.8 | 3060 |
| BPE α = 0 | 1.153 ± 0.001 | 1.109 | 279.2 | 2989 |

- Differences are small (≈ 4 % between best and worst), of the order of seed noise for neighbours; Parity-aware
  BPE's two variants (identical tokenization, permuted token ids) differ by 0.001–0.015 bpb, a direct estimate of
  the noise floor.
- The ranking follows morphological alignment more than compression: the most compressive tokenizer (BPE α = 0)
  gives the worst LM. Caveat: with matched bytes, tokenizers that produce more tokens also get more optimisation
  steps (MorphBPE +7 %, Unigram +17 % vs BPE).
- Fair tokenizers do not equalise the LMs: worst-language bits per FLORES sentence varies by only 2 % across
  tokenizers; the LM cost of a language is dominated by the language, not by its token premium.
- *TBD: LMs for SAA tokenizers.*

## 11. Limitations and next steps

- The split-tree candidate set limits all LP tokenizers; adding external candidates (BPE merges, Unigram pieces)
  as extra tree-independent columns is the most promising next step.
- SAA uses one sample; proper practice would re-estimate the chosen vocabulary on an independent sample and
  report the SAA optimality gap.
- MorphBPE is a reimplementation; Indic gold morphology is small (MorphScore mr: 178 words).
- LM study is small-scale (35 M params, 26–30 M training tokens).

## References

ToaST (Schmidt et al. 2026, arXiv 2605.22705); ConvexTok (Tempus et al. 2026); Parity-aware BPE (Foroutan et al.,
ACL 2026); MorphBPE (Asgari et al., Findings ACL 2026); MUTANT (Rana et al., ACL 2026); MorphScore (Arnett & Bergen
2025; Arnett et al. 2025); FineWeb / FineWeb-2; FLORES+; HiGHS.
