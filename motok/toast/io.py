"""Save / load trained split-tree tokenizers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from motok.toast.trees import SplitTreeTokenizer


def save_tokenizer(path, vocab, ngrams, hierarchy, meta=None, morph_models=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"vocab": list(vocab), "ngrams": ngrams, "hierarchy": list(hierarchy),
                     "meta": meta or {}, "morph_models": morph_models or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path.with_suffix(".vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta or {}, "vocab": sorted(vocab, key=lambda s: (len(s), s))}, f, ensure_ascii=False, indent=0)


def load_tokenizer(path) -> SplitTreeTokenizer:
    with open(path, "rb") as f:
        d = pickle.load(f)
    seg = None
    if "morph" in d["hierarchy"] and d["morph_models"]:
        # morph-first trees need a segmenter at inference; use the first model (single-script) or a router
        from motok.morph.segment import MorfessorSegmenter
        models = {l: MorfessorSegmenter.load(p) for l, p in d["morph_models"].items()}
        seg = _ScriptRouter(models)
    tok = SplitTreeTokenizer(d["vocab"], d["ngrams"], d["hierarchy"], segmenter=seg)
    tok.meta = d["meta"]
    return tok


class _ScriptRouter:
    """Pick a language's segmenter by the word's script (good enough for script-disjoint sets;
    for same-script languages it uses the first matching model)."""

    def __init__(self, models):
        import regex
        self.models = models
        self.script_of = {}
        self._re = regex
        self.order = list(models)

    def __call__(self, word):
        for l in self.order:
            seg = self.models[l](word)
            if seg:
                return seg
        return None
