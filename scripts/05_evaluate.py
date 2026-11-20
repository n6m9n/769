#!/usr/bin/env python3
"""Evaluate all tokenizers in <results_dir>/tokenizers on held-out data.

  python scripts/05_evaluate.py --config configs/smoke.yaml [--filter toast_parity]

Uses data/eval/{lang}.txt (held-out text), data/morph/{lang}.test.tsv (gold, where available)
and data/flores/devtest (parallel, if downloaded) for token premiums.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motok.tokload import is_tokenizer_file, load_any
from motok.config import compute_kappa, load_cfg, read_lines
from motok.eval.metrics import evaluate
from motok.morph.segment import load_gold, load_morphscore
from motok.eval.metrics import morph_alignment
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--filter", default="")
ap.add_argument("--max-eval-lines", type=int, default=5000)
ap.add_argument("--max-gold", type=int, default=3000)
ap.add_argument("--jobs", type=int, default=16)
ap.add_argument("--redo", action="store_true", help="re-evaluate tokenizers already in eval.json")
a = ap.parse_args()
cfg = load_cfg(a.config)

eval_texts = {l: read_lines(cfg.data / "eval" / f"{l}.txt", a.max_eval_lines) for l in cfg.langs}
gold = {}
for l in cfg.langs:
    p = cfg.data / "morph" / f"{l}.test.tsv"
    if p.exists():
        gold[l] = load_gold(p, max_items=a.max_gold)
ms_gold = {}
for l in cfg.langs:
    p = cfg.data / "morphscore" / f"{l}.csv"
    if p.exists():
        ms_gold[l] = load_morphscore(p, max_items=a.max_gold)
fl = cfg.data / "flores" / "devtest"
parallel = {l: read_lines(fl / f"{l}.txt") for l in cfg.langs} if (fl / "en.txt").exists() else None
kappa = compute_kappa(cfg)

out_path = cfg.out / "eval.json"
results = json.load(open(out_path)) if out_path.exists() else {}


def run_one(p):
    tok = load_any(p)
    t0 = time.time()
    res = evaluate(tok, eval_texts, gold=gold, parallel=parallel, kappa=kappa)
    # round-trip sanity check on a few lines
    if ms_gold:   # MorphScore v2 gold (UD words), all six languages
        res["morph_ms"] = {l: morph_alignment(tok, g) for l, g in ms_gold.items()}
        res["ms_f1_mean"] = float(np.mean([m["f1"] for m in res["morph_ms"].values()]))
        res["ms_spurious_mean"] = float(np.mean([m["spurious_per_word"] for m in res["morph_ms"].values()]))
        res["ms_recall_mean"] = float(np.mean([m["morphscore"] for m in res["morph_ms"].values()]))
    res["roundtrip_failures"] = sum(tok.decode_tokens(t) != t for ts in eval_texts.values() for t in ts[:20])
    res["eval_seconds"] = time.time() - t0
    return p.stem, res


paths = [p for p in sorted((cfg.out / "tokenizers").iterdir())
         if a.filter in p.name and is_tokenizer_file(p) and (a.redo or p.stem not in results)]
from multiprocessing import Pool
with Pool(max(1, min(a.jobs, len(paths) or 1))) as pool:
    for name, res in pool.imap_unordered(run_one, paths):
        results[name] = res
        print(f"{name:<26} T_mean={res['T_mean']:7.2f} T_max={res['T_max']:7.2f} gini={res['T_gini']:.3f} "
              f"renyi={res['renyi_eff']:.3f} morphF1={res.get('morph_f1_mean', float('nan')):.3f} "
              f"spur/word={res.get('spurious_mean', float('nan')):.3f} msF1={res.get('ms_f1_mean', float('nan')):.3f} roundtrip_bad={res['roundtrip_failures']} "
              f"({res['eval_seconds']:.0f}s)", flush=True)
        json.dump(results, open(out_path, "w"), indent=1, default=float)
print(f"-> {out_path}")
