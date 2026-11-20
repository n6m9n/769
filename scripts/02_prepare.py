#!/usr/bin/env python3
"""Count pretokens, train Morfessor per language, build the split-tree problem.

  python scripts/02_prepare.py --config configs/smoke.yaml
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motok.config import compute_kappa, load_cfg
from motok.morph.segment import MorfessorSegmenter, train_morfessor, words_from_pretokens
from motok.pretokenize import count_pretokens_file, load_counts, save_counts
from motok.toast.problem import build_problem

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
ap.add_argument("--max-lines", type=int, default=None)
ap.add_argument("--procs", type=int, default=32)
ap.add_argument("--trees-per-lang", type=int, default=None, help="override config")
ap.add_argument("--out-name", default=None, help="write problem to results/<out-name> instead of the config results_dir")
a = ap.parse_args()
cfg = load_cfg(a.config)
if a.trees_per_lang: cfg["trees_per_lang"] = a.trees_per_lang
if a.out_name: cfg["results_dir"] = f"results/{a.out_name}"

def _morfessor_job(args):
    l, cp, mp, max_words = args
    t0 = time.time()
    c, _ = load_counts(cp)
    model = train_morfessor(words_from_pretokens(c), max_words=max_words)
    MorfessorSegmenter(model).save(mp)
    return l, time.time() - t0


counts, segs, morph_paths = {}, {}, {}
for l in cfg.langs:   # counting: each language uses all cores via line chunks
    cp = cfg.data / "counts" / f"{l}.json"
    if cp.exists():
        counts[l], st = load_counts(cp)
    else:
        t0 = time.time()
        counts[l], st = count_pretokens_file(cfg.data / "raw" / f"{l}.txt", a.max_lines, n_proc=a.procs)
        save_counts(cp, counts[l], st)
        print(f"[count] {l}: {st['chars']/1e6:.1f}M chars, {len(counts[l])} pretoken types ({time.time()-t0:.0f}s)", flush=True)
todo = [(l, cfg.data / "counts" / f"{l}.json", cfg.data / "morph_models" / f"{l}.pkl", cfg.morfessor_max_words or 50000)
        for l in cfg.langs if not (cfg.data / "morph_models" / f"{l}.pkl").exists()]
if todo:   # Morfessor: one process per language
    from multiprocessing import Pool
    with Pool(len(todo)) as pool:
        for l, sec in pool.imap_unordered(_morfessor_job, todo):
            print(f"[morfessor] {l}: trained ({sec:.0f}s)", flush=True)
for l in cfg.langs:
    mp = cfg.data / "morph_models" / f"{l}.pkl"
    segs[l] = MorfessorSegmenter.load(mp)
    morph_paths[l] = str(mp)

kappa = compute_kappa(cfg)
print("[kappa]", {l: round(k, 3) for l, k in kappa.items()})
t_build = time.time()
prob = build_problem(counts, cfg.trees_per_lang, hierarchy=cfg.hierarchy, segmenters=segs, kappa=kappa,
                     balance_ngrams=cfg.balance_ngrams)
prob.meta["morph_models"] = morph_paths
prob.meta["build_seconds"] = time.time() - t_build
prob.save(cfg.out / "problem.pkl")
json.dump({"coverage": prob.coverage, "kappa": kappa, "n_trees": len(prob.trees),
           "n_nodes": int(sum(t.n_nodes for t in prob.trees)), "base_chars": len(prob.base_chars), "trees_per_lang": cfg.trees_per_lang,
           "build_seconds": prob.meta["build_seconds"]},
          open(cfg.out / "problem_stats.json", "w"), indent=1)
print(f"[prepare] saved {cfg.out/'problem.pkl'}")
