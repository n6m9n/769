#!/usr/bin/env python3
"""Train frequency-based baselines: {BPE, Unigram} x temperature alpha.

  python scripts/03_baselines.py --config configs/smoke.yaml
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motok.baselines.hf import temperature_mixture, train_hf
from motok.config import load_cfg

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
cfg = load_cfg(a.config)
files = {l: str(cfg.data / "raw" / f"{l}.txt") for l in cfg.langs}
for kind in cfg.baselines["kinds"]:
    for alpha in cfg.baselines["alphas"]:
        out = cfg.out / "tokenizers" / f"{kind}_a{alpha}.json"
        if out.exists():
            continue
        t0 = time.time()
        train_hf(kind, temperature_mixture(files, alpha), cfg.vocab_size, str(out))
        print(f"[baseline] {kind} alpha={alpha} ({time.time()-t0:.0f}s) -> {out.name}")
