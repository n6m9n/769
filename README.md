# motok — Multi-objective optimisation for multilingual tokenization

Vocabulary selection as a **multi-objective linear program** over split trees (extends ToaST,
Schmidt et al. 2026). Objectives: compression, cross-lingual parity (worst-language cost),
morphological alignment (spurious morpheme boundaries), cross-lingual sharing. Pareto fronts are
traced with warm-started ε-constraint sweeps; reduced costs give the exchange rate between objectives.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q                      # 9 tests, < 5 s
```

## Pipeline
Every script takes `--config configs/smoke.yaml` (UD treebanks, minutes on a laptop) or
`configs/full.yaml` (FineWeb-2, needs real compute).

| step | script | output |
|---|---|---|
| 1 | `01_download.py --what smoke` / `--what fineweb` / `--what flores` | `data/raw`, `data/eval`, `data/morph`, `data/flores` |
| 2 | `02_prepare.py` | pretoken counts, Morfessor models, `results/<name>/problem.pkl` |
| 3 | `03_baselines.py` | BPE / Unigram × temperature α in `results/<name>/tokenizers/` |
| 4 | `04_sweep.py` | anchors + parity / morph / grid sweeps → `sweep.json`, one tokenizer per point |
| 5 | `05_evaluate.py` | held-out metrics for every tokenizer → `eval.json` |
| 6 | `06_plots.py` | Pareto fronts, per-language bars, shadow prices, `summary.md` |

FLORES+ is gated: accept the terms at https://huggingface.co/datasets/openlanguagedata/flores_plus
and run `huggingface-cli login`. FLORES is used for parity metrics and for the content
normalisation κ_l (chars of language l per English char on parallel text). Without it κ = 1.

## Layout
```
motok/pretokenize.py     GPT-4o length-limited regex (keeps Indic marks attached), counting
motok/toast/trees.py     n-gram counts, split trees (morph > grapheme > char hierarchy), inference, tokenizer
motok/toast/problem.py   multilingual problem: tree selection per language, balanced n-grams, morph marks
motok/lp/model.py        VocabLP: the multi-objective LP (HiGHS), scalarisation, warm starts, rounding, exact eval
motok/morph/segment.py   Morfessor (training signal), SIGMORPHON gold loader (evaluation only)
motok/baselines/hf.py    HF BPE/Unigram with the same pretokenizer + temperature sampling
motok/eval/metrics.py    compression, Gini, Rényi efficiency, morph P/R/F1 + MorphScore, sharing, hypervolume
```

## Preliminary smoke results (UD text, 6 languages, 10.7k trees, vocab 4000)
See `results_preview/sweep.log`. Compression-only anchor is fully integral; the parity sweep cuts the
worst language's tokens/1000 chars by 31% (633 → 437) for +2.7% total tokens; all integrality gaps ≤ 4e-4.
Caveats: Marathi/Tamil have ~350 training sentences; κ = 1 (no FLORES); held-out eval not yet run.

## Known issues / TODO
- Morph ε-sweep: warm dual simplex stalled once; `solve()` now has a 60 s warm-start limit with cold fallback. Verify.
- Leaf-coverage rows are built in a Python loop — vectorise before scaling past ~100k trees.
- `ngram_counts` holds all substrings in memory — chunk/prune for full FineWeb data.
- External baselines to add: Parity-aware BPE (github.com/swiss-ai/parity-aware-bpe), MorphBPE (github.com/qcri/MorphBPE).
- Gold morphology for Indic languages: check MorphScore (github.com/catherinearnett/morphscore) coverage.
- Small-LM downstream evaluation (bits-per-byte, byte-matched training) not started.
