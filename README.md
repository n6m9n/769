# motok — Multi-objective optimisation for multilingual tokenization

Vocabulary selection as a **multi-objective linear program** over split trees (extends ToaST,
Schmidt et al. 2026). Objectives: compression, cross-lingual parity (worst-language cost),
morphological alignment (spurious morpheme boundaries), cross-lingual sharing. Pareto fronts are
traced with warm-started ε-constraint sweeps; reduced costs give the exchange rate between objectives.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q                      # 13 tests, < 5 s
```

## Pipeline
Every script takes `--config configs/smoke.yaml` (UD treebanks, minutes on a laptop) or
`configs/full.yaml` (FineWeb / FineWeb-2, ~500 MB, needs a server).

| step | script | output |
|---|---|---|
| 1 | `01_download.py --what smoke\|fineweb\|flores [--data-dir data_full]` | raw / eval text, FLORES+, SIGMORPHON gold |
| 2 | `02_prepare.py [--trees-per-lang K] [--tail-samples S] [--hierarchy morph grapheme char] [--out-name NAME]` | pretoken counts, Morfessor, `results/<name>/problem.pkl` |
| 3 | `03_baselines.py --kinds bpe unigram parity_bpe morph_bpe` | BPE / Unigram × temperature α, Parity-aware BPE, MorphBPE |
| 4 | `04_sweep.py` (small) or `scripts/run_lp.sh <cfg> <name>` (parallel: anchors → sweeps → merge) | `sweep.json`, one tokenizer per Pareto point |
| 5 | `05_evaluate.py` | held-out metrics (FineWeb eval, FLORES+, MorphScore + SIGMORPHON gold) → `eval.json` |
| 6 | `07_lm.py` / `scripts/run_lm_queue.sh <cfg> <gpu> "<seeds>" tok...` | small GPT per tokenizer, bits per byte |
| 7 | `08_report.py --lp <dir>:<prefix> --others ...` | `results/final/` figures + tables |

`scripts/link_lp.sh <run> <prefix>` links a run's LP tokenizers into `results/full/tokenizers` so every tokenizer
is evaluated and trained on in one place. `check_results.py` verifies LP ≤ integer, bound feasibility, monotone
fronts and round trips.

Key options of the LP:
- `--tail-samples S` (02): sample average approximation of the pretoken tail; with `--tail-tokens head` (04, default)
  tail trees only reweight the objective and cannot add candidate tokens (avoids the SAA optimizer's curse).
- bound shading (04, automatic): if rounding overshoots an ε-bound, the bound is tightened and the LP re-solved, so
  every reported gap is a certificate.
- `--warm-anchor` (04): solve min-max / min-morph anchors warm from the compression basis (primal simplex).

FLORES+ is gated: accept the terms at https://huggingface.co/datasets/openlanguagedata/flores_plus
and run `huggingface-cli login`. MorphScore gold: `catherinearnett/morphscore` on Hugging Face (`data/morphscore/`).

## Layout
```
motok/pretokenize.py     GPT-4o length-limited regex (keeps Indic marks attached), counting
motok/toast/trees.py     n-gram counts, split trees (morph > grapheme > char hierarchy), inference, tokenizer
motok/toast/problem.py   multilingual problem: tree selection per language, balanced n-grams, morph marks
motok/lp/model.py        VocabLP: the multi-objective LP (HiGHS), scalarisation, warm starts, rounding, exact eval
motok/morph/segment.py   Morfessor (training signal), SIGMORPHON gold loader (evaluation only)
motok/baselines/hf.py    HF BPE/Unigram with the same pretokenizer + temperature sampling
motok/baselines/parity_bpe.py  Parity-aware BPE (authors' learner in ext/, our pretokenizer)
motok/baselines/morph_bpe.py   MorphBPE reimplementation (no merges across Morfessor boundaries)
motok/tokload.py         one loader for every tokenizer type
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
