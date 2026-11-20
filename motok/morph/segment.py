"""Morphological segmentation.

Training signal: unsupervised Morfessor Baseline per language (works for every language,
including Indic ones without gold data). Evaluation: gold segmentations only
(SIGMORPHON 2022 / MorphScore), so we never evaluate on the same signal we optimise.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import morfessor
import regex

_WORD = regex.compile(r"^[\p{L}\p{M}]+$")


def train_morfessor(word_counts: Counter, max_words: int = 50_000, type_based: bool = True, seed: int = 0):
    """Train Morfessor Baseline on the most frequent letter-only words."""
    import random
    random.seed(seed)
    words = [(c, w) for w, c in word_counts.most_common() if _WORD.match(w)][:max_words]
    model = morfessor.BaselineModel()
    model.load_data(words, count_modifier=(lambda x: 1) if type_based else None)
    model.train_batch()
    return model


def words_from_pretokens(pretoken_counts: Counter) -> Counter:
    """Strip leading space/punct from pretokens and keep letter-only bodies."""
    out: Counter = Counter()
    for p, c in pretoken_counts.items():
        m = regex.search(r"[\p{L}\p{M}]+$", p)
        if m and m.start() <= 1:
            out[m.group(0)] += c
    return out


class MorfessorSegmenter:
    def __init__(self, model):
        self.model = model
        self._cache: dict[str, list[str]] = {}

    def __call__(self, word: str) -> list[str] | None:
        if not word or not _WORD.match(word):
            return None
        hit = self._cache.get(word)
        if hit is None:
            hit = list(self.model.viterbi_segment(word)[0])
            self._cache[word] = hit
        return hit

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))


def load_gold(path: str | Path, surface_only: bool = True, max_items: int | None = None) -> list[tuple[str, list[str]]]:
    """SIGMORPHON 2022 word-level format: word<TAB>m1 @@m2 @@m3<TAB>category.

    Some segmentations are canonical (morphemes don't concatenate to the surface word);
    with surface_only=True those are dropped so boundaries are well defined.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word, segs = parts[0], [s.strip() for s in parts[1].split("@@")]
            segs = [s for s in segs if s]
            if surface_only and "".join(segs) != word:
                continue
            out.append((word, segs))
            if max_items and len(out) >= max_items:
                break
    return out


def boundaries_of(segs: list[str]) -> set[int]:
    pos, out = 0, set()
    for s in segs[:-1]:
        pos += len(s)
        out.add(pos)
    return out


def load_morphscore(path: str | Path, max_items: int | None = 3000, seed: int = 0) -> list[tuple[str, list[str]]]:
    """MorphScore v2 (Arnett et al. 2025) gold from UD: word = [preceding] + stem + [following].

    Default MorphScore filtering: unique items, stem == lemma, no digits; segments must concatenate to the
    word form. Words with a single morpheme are kept (they test over-segmentation). A uniform random
    sample of `max_items` distinct word forms (fixed seed) keeps evaluation fast.
    """
    import csv
    import random
    out, seen = [], set()
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            w = row["wordform"]
            if row.get("unique") != "unique" or row["lemma"] != row["stem"] or w in seen:
                continue
            if regex.search(r"\d", w) or not w:
                continue
            segs = [s for s in (row.get("preceding_part") or "", row["stem"], row.get("following_part") or "") if s]
            if "".join(segs) != w:
                continue
            seen.add(w)
            out.append((w, segs))
    if max_items and len(out) > max_items:
        out = random.Random(seed).sample(out, max_items)
    return out
