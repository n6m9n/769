#!/usr/bin/env python3
"""Train baselines (same pretokenizer and vocab size as the LP tokenizers).

  python scripts/03_baselines.py --config configs/full.yaml [--kinds bpe unigram parity_bpe morph_bpe] [--jobs 4]

  bpe / unigram   HF byte-level BPE / Unigram x temperature alpha (config baselines.alphas)
  parity_bpe      Parity-aware BPE (Foroutan et al. 2026), authors' learner, FLORES+ dev as parity set
                  (variants: base, window)
  morph_bpe       MorphBPE (Asgari et al. 2026), reimplemented: no merges across Morfessor boundaries
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motok.baselines.hf import temperature_mixture, train_hf
from motok.config import load_cfg
from motok.pretokenize import load_counts

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--kinds", nargs="+", default=None)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--parity-min-count", type=int, default=3)
a = ap.parse_args()
cfg = load_cfg(a.config)
tokdir = cfg.out / "tokenizers"
tokdir.mkdir(parents=True, exist_ok=True)
files = {l: str(cfg.data / "raw" / f"{l}.txt") for l in cfg.langs}
kinds = a.kinds or cfg.baselines["kinds"]


def job(spec):
    kind, arg = spec
    t0 = time.time()
    if kind in ("bpe", "unigram"):
        out = tokdir / f"{kind}_a{arg}.json"
        train_hf(kind, temperature_mixture(files, arg), cfg.vocab_size, str(out))
    elif kind == "parity_bpe":
        from motok.baselines.parity_bpe import train_parity_bpe
        out = tokdir / f"paritybpe_{arg}.json"
        counts = {l: load_counts(cfg.data / "counts" / f"{l}.json")[0] for l in cfg.langs}
        dev = {l: str(cfg.data / "flores" / "dev" / f"{l}.txt") for l in cfg.langs}
        train_parity_bpe(counts, dev, cfg.vocab_size, str(out), variant=arg, min_count=a.parity_min_count)
    elif kind == "morph_bpe":
        from motok.baselines.morph_bpe import MorphMarker, train_joint_morfessor, train_morph_bpe
        from motok.morph.segment import MorfessorSegmenter, words_from_pretokens
        out = tokdir / "morphbpe.json"
        mp = tokdir / "morphbpe.morf.pkl"
        if not mp.exists():
            wc = {l: words_from_pretokens(load_counts(cfg.data / "counts" / f"{l}.json")[0]) for l in cfg.langs}
            MorfessorSegmenter(train_joint_morfessor(wc, cfg.morfessor_max_words or 20000)).save(mp)
        marker = MorphMarker(MorfessorSegmenter.load(mp))
        train_morph_bpe(temperature_mixture(files, 1.0), marker, cfg.vocab_size, str(out))
    else:
        raise ValueError(kind)
    return f"[baseline] {kind} {arg} ({time.time()-t0:.0f}s) -> {out.name}"


specs = []
for kind in kinds:
    if kind in ("bpe", "unigram"):
        specs += [(kind, al) for al in cfg.baselines["alphas"] if not (tokdir / f"{kind}_a{al}.json").exists()]
    elif kind == "parity_bpe":
        specs += [(kind, v) for v in ["base", "window"] if not (tokdir / f"paritybpe_{v}.json").exists()]
    elif kind == "morph_bpe" and not (tokdir / "morphbpe.json").exists():
        specs.append((kind, None))
# slowest first
order = {"parity_bpe": 0, "unigram": 1, "morph_bpe": 2, "bpe": 3}
specs.sort(key=lambda s: order[s[0]])
print(f"[baselines] {len(specs)} to train: {specs}", flush=True)
with Pool(max(1, min(a.jobs, len(specs)))) as pool:
    for msg in pool.imap_unordered(job, specs):
        print(msg, flush=True)
