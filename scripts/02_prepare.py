#!/usr/bin/env python3
"""Count pretokens, train Morfessor per language, build the split-tree problem.

  python scripts/02_prepare.py --config configs/smoke.yaml
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motok.config import compute_kappa, load_cfg
from motok.morph.segment import MorfessorSegmenter, train_morfessor, words_from_pretokens
from motok.pretokenize import count_pretokens, iter_lines, load_counts, save_counts
from motok.toast.problem import build_problem

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
ap.add_argument("--max-lines", type=int, default=None)
a = ap.parse_args()
cfg = load_cfg(a.config)

counts, segs, morph_paths = {}, {}, {}
for l in cfg.langs:
    cp = cfg.data / "counts" / f"{l}.json"
    if cp.exists():
        counts[l], st = load_counts(cp)
    else:
        t0 = time.time()
        counts[l], st = count_pretokens(iter_lines(cfg.data / "raw" / f"{l}.txt", a.max_lines))
        save_counts(cp, counts[l], st)
        print(f"[count] {l}: {st['chars']/1e6:.1f}M chars, {len(counts[l])} pretoken types ({time.time()-t0:.0f}s)")
    mp = cfg.data / "morph_models" / f"{l}.pkl"
    if not mp.exists():
        t0 = time.time()
        model = train_morfessor(words_from_pretokens(counts[l]), max_words=cfg.morfessor_max_words or 50000)
        MorfessorSegmenter(model).save(mp)
        print(f"[morfessor] {l}: trained ({time.time()-t0:.0f}s)")
    segs[l] = MorfessorSegmenter.load(mp)
    morph_paths[l] = str(mp)

kappa = compute_kappa(cfg)
print("[kappa]", {l: round(k, 3) for l, k in kappa.items()})
prob = build_problem(counts, cfg.trees_per_lang, hierarchy=cfg.hierarchy, segmenters=segs, kappa=kappa,
                     balance_ngrams=cfg.balance_ngrams)
prob.meta["morph_models"] = morph_paths
prob.save(cfg.out / "problem.pkl")
json.dump({"coverage": prob.coverage, "kappa": kappa, "n_trees": len(prob.trees),
           "n_nodes": int(sum(t.n_nodes for t in prob.trees)), "base_chars": len(prob.base_chars)},
          open(cfg.out / "problem_stats.json", "w"), indent=1)
print(f"[prepare] saved {cfg.out/'problem.pkl'}")
