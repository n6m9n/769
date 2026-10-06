"""Assemble the multilingual split-tree problem that the LP optimises over."""
from __future__ import annotations

import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from motok.toast.trees import (SplitTree, annotate_morph, body_start, build_tree, make_levels,
                               morph_boundaries, ngram_counts)


@dataclass
class Problem:
    langs: list[str]
    trees: list[SplitTree]
    counts: np.ndarray            # [n_trees, n_langs] raw pretoken counts c_{j,l}
    chars: np.ndarray             # [n_langs] characters covered by selected trees (per language)
    kappa: np.ndarray             # [n_langs] content-normalisation factor (FLORES chars_l / chars_en); 1 if unknown
    ngrams: dict                  # n-gram counts used to build trees (needed at inference)
    hierarchy: list[str]
    coverage: dict                # per-language fraction of pretoken mass covered
    base_chars: list[str]         # leaves; always in vocabulary
    meta: dict = field(default_factory=dict)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "Problem":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_problem(
    lang_counts: dict[str, Counter],
    trees_per_lang: int | dict[str, int],
    hierarchy: list[str] = ("grapheme", "char"),
    segmenters: dict | None = None,
    kappa: dict[str, float] | None = None,
    balance_ngrams: bool = True,
    max_pretoken_len: int = 64,
    verbose: bool = True,
) -> Problem:
    """
    lang_counts:     pretoken Counter per language (from the training corpus)
    trees_per_lang:  number of most frequent pretokens per language that become split trees
    hierarchy:       split-position preference, e.g. ["morph","grapheme","char"]
    segmenters:      lang -> callable(word)->segments; used for morph marks (and tree splits if 'morph' in hierarchy)
    balance_ngrams:  rescale each language to equal total mass before computing n-gram counts,
                     so split decisions for low-resource languages aren't starved of counts
    """
    t0 = time.time()
    langs = list(lang_counts)
    L = len(langs)
    segmenters = segmenters or {}
    totals = {l: sum(lang_counts[l].values()) for l in langs}
    mean_total = float(np.mean(list(totals.values())))
    scale = {l: (mean_total / totals[l] if balance_ngrams else 1.0) for l in langs}

    # 1) select pretokens per language
    sel: dict[str, np.ndarray] = {}
    coverage = {}
    for li, l in enumerate(langs):
        K = trees_per_lang[l] if isinstance(trees_per_lang, dict) else trees_per_lang
        top = [(p, c) for p, c in lang_counts[l].most_common() if len(p) <= max_pretoken_len][:K]
        coverage[l] = sum(c for _, c in top) / max(totals[l], 1)
        for p, c in top:
            v = sel.setdefault(p, np.zeros(L))
            v[li] += c
    # add counts of selected pretokens in *other* languages too (shared words, punctuation)
    for p, v in sel.items():
        for li, l in enumerate(langs):
            if v[li] == 0:
                v[li] = lang_counts[l].get(p, 0)
    pretokens = list(sel)
    C = np.stack([sel[p] for p in pretokens])

    # 2) n-gram counts (language-balanced), threshold = min weighted count over selected pretokens
    merged: dict[str, float] = {}
    for l in langs:
        s = scale[l]
        for p, c in lang_counts[l].items():
            if len(p) <= max_pretoken_len:
                merged[p] = merged.get(p, 0.0) + c * s
    min_count = min(merged[p] for p in pretokens)
    if verbose:
        print(f"[problem] {len(pretokens)} trees; n-gram min_count={min_count:.3g}; computing n-grams over {len(merged)} pretoken types")
    ngrams = ngram_counts(merged, min_count=min_count, max_len=max_pretoken_len)

    # 3) trees + morph annotation
    trees = []
    n_morph = 0
    for j, p in enumerate(pretokens):
        mb = None
        b0 = body_start(p)
        body = p[b0:]
        # use the segmenter of the language contributing most counts to this pretoken
        lang = langs[int(np.argmax(C[j]))]
        seg = segmenters.get(lang)
        if seg is not None and body:
            mb = morph_boundaries(p, seg(body))
        if mb is not None:
            n_morph += 1
        t = build_tree(p, ngrams, make_levels(p, list(hierarchy), mb if "morph" in hierarchy else None))
        annotate_morph(t, mb)
        t.meta["has_morph"] = mb is not None
        trees.append(t)

    chars = (C * np.array([len(p) for p in pretokens])[:, None]).sum(0)
    kap = np.array([(kappa or {}).get(l, 1.0) for l in langs])
    base_chars = sorted({ch for p in pretokens for ch in p})
    if verbose:
        n_nodes = sum(t.n_nodes for t in trees)
        print(f"[problem] {n_nodes} nodes, {len(base_chars)} base chars, morph-annotated trees: {n_morph}/{len(trees)}, "
              f"coverage: {', '.join(f'{l}={coverage[l]:.1%}' for l in langs)}  ({time.time()-t0:.1f}s)")
    return Problem(langs, trees, C, chars, kap, ngrams, list(hierarchy), coverage, base_chars,
                   meta={"min_count": min_count, "trees_per_lang": trees_per_lang, "balance_ngrams": balance_ngrams})
