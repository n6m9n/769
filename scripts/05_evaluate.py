#!/usr/bin/env python3
"""Evaluate all tokenizers in <results_dir>/tokenizers on held-out data.

  python scripts/05_evaluate.py --config configs/smoke.yaml [--filter toast_parity]

Uses data/eval/{lang}.txt (held-out text), data/morph/{lang}.test.tsv (gold, where available)
and data/flores/devtest (parallel, if downloaded) for token premiums.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motok.baselines.hf import HFWrapper, SplitTreeWrapper
from motok.config import compute_kappa, load_cfg, read_lines
from motok.eval.metrics import evaluate
from motok.morph.segment import load_gold
from motok.toast.io import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--filter", default="")
ap.add_argument("--max-eval-lines", type=int, default=5000)
ap.add_argument("--max-gold", type=int, default=3000)
a = ap.parse_args()
cfg = load_cfg(a.config)

eval_texts = {l: read_lines(cfg.data / "eval" / f"{l}.txt", a.max_eval_lines) for l in cfg.langs}
gold = {}
for l in cfg.langs:
    p = cfg.data / "morph" / f"{l}.test.tsv"
    if p.exists():
        gold[l] = load_gold(p, max_items=a.max_gold)
fl = cfg.data / "flores" / "devtest"
parallel = {l: read_lines(fl / f"{l}.txt") for l in cfg.langs} if (fl / "en.txt").exists() else None
kappa = compute_kappa(cfg)

out_path = cfg.out / "eval.json"
results = json.load(open(out_path)) if out_path.exists() else {}
for p in sorted((cfg.out / "tokenizers").iterdir()):
    if a.filter not in p.name:
        continue
    if p.suffix == ".json" and not p.name.endswith(".vocab.json"):
        tok = HFWrapper.load(p, p.stem)
    elif p.suffix == ".pkl":
        tok = SplitTreeWrapper(load_tokenizer(p), p.stem)
    else:
        continue
    t0 = time.time()
    res = evaluate(tok, eval_texts, gold=gold, parallel=parallel, kappa=kappa)
    # round-trip sanity check on a few lines
    bad = sum(tok.decode_tokens(t) != t for ts in eval_texts.values() for t in ts[:20])
    res["roundtrip_failures"] = bad
    results[p.stem] = res
    print(f"{p.stem:<26} T_mean={res['T_mean']:7.2f} T_max={res['T_max']:7.2f} gini={res['T_gini']:.3f} "
          f"renyi={res['renyi_eff']:.3f} morphF1={res.get('morph_f1_mean', float('nan')):.3f} "
          f"spur/word={res.get('spurious_mean', float('nan')):.3f} roundtrip_bad={bad} ({time.time()-t0:.0f}s)")
json.dump(results, open(out_path, "w"), indent=1, default=float)
print(f"-> {out_path}")
