# Context for Claude Code

Course project (Optimization in ML): multi-objective optimisation for multilingual tokenization.
Core idea: ToaST's vocabulary-selection LP is linear in node variables z, so extra objectives
(per-language cost, worst-language cost, spurious morpheme boundaries, sharing) are added as
*aggregate columns* defined by equality rows. Scalarisations only change costs/bounds on those
columns, so sweeps warm-start, and reduced costs = shadow prices (exchange rates).

## Conventions
- Units: T_l = tokens per 1000 (English-equivalent) chars; M_l = spurious boundaries per 1000 chars.
  Ttot / Mtot = data-share weighted. Lower is better for everything except A (aligned) and S (sharing).
- Trees are character-level (never split inside a UTF-8 char); 256 byte-fallback tokens are reserved
  outside the LP (`m_LP = vocab_size - 256`).
- Morfessor segmentations are the *training* signal; gold SIGMORPHON/MorphScore is *evaluation only*.
  Never evaluate morphology on Morfessor output.
- Cold solves (after objective changes) use presolve: `lp.solve(warm=False)`. Bound/RHS sweeps use
  `warm=True, method="dual"`. Sweep bounds from loose to tight.

## Invariants (keep `pytest -q` green)
- If the LP solution is integral, `lp.evaluate(lp.round(res))` must equal the LP aggregates exactly.
- Rounded vocab objective ≥ LP objective (LP is a lower bound).
- Tokenizer decode(encode(s)) == s, including unseen characters (byte fallback).

## Commands
pytest -q
python scripts/01_download.py --what smoke
python scripts/02_prepare.py --config configs/smoke.yaml
python scripts/03_baselines.py --config configs/smoke.yaml
python scripts/04_sweep.py --config configs/smoke.yaml
python scripts/05_evaluate.py --config configs/smoke.yaml
python scripts/06_plots.py --config configs/smoke.yaml

## Priorities
1. Run the smoke pipeline end to end; confirm the morph sweep no longer stalls.
2. Full data: start `configs/full.yaml` at 5k trees/language, record build + solve times, then scale.
3. Add Parity-aware BPE and MorphBPE baselines; compare fronts with hypervolume.
4. Small-LM downstream eval (bits-per-byte) on 4–6 selected tokenizers.
