"""Config + path helpers shared by the scripts."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Cfg(dict):
    __getattr__ = dict.get

    @property
    def data(self) -> Path:
        return ROOT / self["data_dir"]

    @property
    def out(self) -> Path:
        p = ROOT / self["results_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_cfg(path) -> Cfg:
    return Cfg(yaml.safe_load(open(path)))


def read_lines(path, max_lines=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if max_lines and i >= max_lines:
                break
            l = l.rstrip("\n")
            if l:
                out.append(l)
    return out


def compute_kappa(cfg: Cfg, ref="en", split="dev") -> dict:
    """kappa_l = chars of FLORES text in l / chars of the same content in English (1.0 if FLORES missing)."""
    d = cfg.data / "flores" / split
    if not (d / f"{ref}.txt").exists():
        return {l: 1.0 for l in cfg.langs}
    ref_chars = sum(len(x) for x in read_lines(d / f"{ref}.txt"))
    return {l: (sum(len(x) for x in read_lines(d / f"{l}.txt")) / ref_chars) if (d / f"{l}.txt").exists() else 1.0
            for l in cfg.langs}
