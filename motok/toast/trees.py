"""Split trees (ToaST, Schmidt et al. 2026) at character level with a preference hierarchy.

A split tree for a pretoken is a full binary tree whose leaves are single characters.
Each internal node is split at the position maximizing min(count(left), count(right)),
choosing among allowed positions of the highest non-empty preference level
(e.g. morpheme boundaries > grapheme-cluster boundaries > any character boundary).
This is the hierarchy sketched in ToaST Appendix A.

Trees depend only on n-gram counts (+ optional morph boundaries), never on the
vocabulary, so the vocabulary LP can treat them as fixed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import regex

_GRAPHEME = regex.compile(r"\X")
_LETTER = regex.compile(r"[\p{L}\p{M}]")


# ---------------------------------------------------------------- n-gram counts
def ngram_counts(pretoken_counts: dict[str, float], min_count: float, max_len: int = 64) -> dict[str, float]:
    """Character n-gram counts within pretokens, weighted by pretoken count.

    Keeps n-grams with count >= min_count. Substring counts are monotone
    (a substring is at least as frequent as any string containing it), so if
    min_count <= the count of every selected pretoken, every substring of every
    selected pretoken is in the dictionary.
    """
    acc: dict[str, float] = {}
    get = acc.get
    for p, c in pretoken_counts.items():
        L = len(p)
        for i in range(L):
            for j in range(i + 1, min(L, i + max_len) + 1):
                s = p[i:j]
                acc[s] = get(s, 0.0) + c
    return {s: c for s, c in acc.items() if c >= min_count}


# ---------------------------------------------------------------- boundaries
def grapheme_boundaries(s: str) -> set[int]:
    pos, out = 0, set()
    for g in _GRAPHEME.findall(s):
        pos += len(g)
        out.add(pos)
    out.discard(len(s))
    return out


def body_start(s: str) -> int:
    """Index of the first letter/mark (skips a leading space or punctuation char)."""
    m = _LETTER.search(s)
    return m.start() if m else 0


def morph_boundaries(s: str, segments: list[str] | None) -> set[int] | None:
    """Global morpheme-boundary positions for pretoken `s` given segments of its body.

    Returns None if the segmentation does not match the surface string. The body
    start (e.g. after a leading space) is always a boundary.
    """
    if segments is None:
        return None
    b0 = body_start(s)
    body = s[b0:]
    if "".join(segments) != body[: sum(map(len, segments))] or not segments:
        return None
    out, pos = {b0} if b0 > 0 else set(), b0
    for seg in segments[:-1]:
        pos += len(seg)
        out.add(pos)
    out.discard(0)
    out.discard(len(s))
    return out


# ---------------------------------------------------------------- tree
@dataclass
class SplitTree:
    """Nodes stored in preorder. node 0 is the root."""
    text: str
    start: np.ndarray       # [n_nodes]
    end: np.ndarray         # [n_nodes]
    left: np.ndarray        # child index or -1
    right: np.ndarray
    parent: np.ndarray      # parent index or -1
    misaligned: np.ndarray | None = None  # [n_nodes] 1 if node starts strictly inside a morpheme
    aligned: np.ndarray | None = None     # [n_nodes] 1 if node starts at an interior morph boundary
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.start)

    def token(self, k: int) -> str:
        return self.text[self.start[k]: self.end[k]]

    def leaves(self) -> np.ndarray:
        return np.nonzero(self.left < 0)[0]

    def path_to_root(self, k: int) -> list[int]:
        out = []
        while k >= 0:
            out.append(k)
            k = self.parent[k]
        return out


def _score(counts, a: str, b: str) -> float:
    return min(counts.get(a, -1.0), counts.get(b, -1.0))


def best_split(s: str, offset: int, counts: dict, levels: list[set[int] | None]) -> int:
    """Return split position (relative to s) using the preference hierarchy.

    levels: list of sets of *global* allowed positions; None means 'all positions'.
    """
    L = len(s)
    for allowed in levels:
        if allowed is None:
            cand = range(1, L)
        else:
            cand = [p - offset for p in sorted(allowed) if offset < p < offset + L]
        best_i, best = -1, -1.0
        for i in cand:
            sc = _score(counts, s[:i], s[i:])
            if sc > best:
                best, best_i = sc, i
        if best_i > 0 and best >= 0:
            return best_i
    # ToaST fallback: split off the longest known prefix
    for i in range(L - 1, 0, -1):
        if s[:i] in counts:
            return i
    return 1


def build_tree(text: str, counts: dict, levels: list[set[int] | None]) -> SplitTree:
    starts, ends, lefts, rights, parents = [], [], [], [], []

    def rec(s0: int, e0: int, par: int) -> int:
        k = len(starts)
        starts.append(s0); ends.append(e0); parents.append(par); lefts.append(-1); rights.append(-1)
        if e0 - s0 > 1:
            i = best_split(text[s0:e0], s0, counts, levels)
            lefts[k] = rec(s0, s0 + i, k)
            rights[k] = rec(s0 + i, e0, k)
        return k

    rec(0, len(text), -1)
    return SplitTree(text, np.array(starts), np.array(ends), np.array(lefts), np.array(rights), np.array(parents))


def make_levels(text: str, hierarchy: list[str], mb: set[int] | None) -> list[set[int] | None]:
    levels: list[set[int] | None] = []
    for h in hierarchy:
        if h == "morph":
            if mb:
                levels.append(mb)
        elif h == "grapheme":
            levels.append(grapheme_boundaries(text))
        elif h == "char":
            levels.append(None)
        else:
            raise ValueError(h)
    if not levels or levels[-1] is not None:
        levels.append(None)
    return levels


def annotate_morph(tree: SplitTree, mb: set[int] | None) -> None:
    """Mark nodes whose start position is a spurious (inside-morpheme) or correct boundary."""
    n = tree.n_nodes
    mis, ali = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    if mb is not None:
        b0 = body_start(tree.text)
        for k in range(n):
            s = tree.start[k]
            if s == 0 or s <= b0:   # word start / leading space-punct region is never "inside a morpheme"
                continue
            if s in mb:
                ali[k] = 1
            else:
                mis[k] = 1
    tree.misaligned, tree.aligned = mis, ali


# ---------------------------------------------------------------- inference
def tree_tokenize(tree: SplitTree, vocab: set[str]) -> list[int]:
    """Indices of emitted nodes: first in-vocabulary node on every root-to-leaf path."""
    out, stack = [], [0]
    while stack:
        k = stack.pop()
        if tree.left[k] < 0 or tree.token(k) in vocab:
            out.append(k)
        else:
            stack.append(tree.right[k]); stack.append(tree.left[k])
    return out


class SplitTreeTokenizer:
    """Inference for a trained vocabulary. Characters outside the vocab fall back to UTF-8 bytes."""

    def __init__(self, vocab: list[str], counts: dict, hierarchy: list[str], segmenter=None):
        self.vocab = list(vocab)
        self.vset = set(vocab)
        self.counts = counts
        self.hierarchy = hierarchy
        self.segmenter = segmenter  # callable(body:str)->list[str]; only needed for 'morph' in hierarchy
        self._cache: dict[str, list[tuple[str, int, int]]] = {}
        self.byte_tokens = [f"<0x{b:02X}>" for b in range(256)]

    def _tokenize_pretoken(self, p: str) -> list[tuple[str, int, int]]:
        """Returns (token, local_start, local_end); byte-fallback pieces share their char's span."""
        hit = self._cache.get(p)
        if hit is not None:
            return hit
        if p in self.vset:
            toks = [(p, 0, len(p))]
        else:
            mb = None
            if "morph" in self.hierarchy and self.segmenter is not None:
                b0 = body_start(p)
                mb = morph_boundaries(p, self.segmenter(p[b0:])) if p[b0:] else None
            tree = build_tree(p, self.counts, make_levels(p, self.hierarchy, mb))
            toks = []
            for k in tree_tokenize(tree, self.vset):
                t, s0, e0 = tree.token(k), int(tree.start[k]), int(tree.end[k])
                if t in self.vset:
                    toks.append((t, s0, e0))
                else:  # unseen single character -> UTF-8 bytes
                    toks.extend((self.byte_tokens[b], s0, e0) for b in t.encode("utf-8"))
        if len(self._cache) < 2_000_000:
            self._cache[p] = toks
        return toks

    def encode(self, text: str) -> list[str]:
        from motok.pretokenize import pretokenize
        out: list[str] = []
        for p in pretokenize(text):
            out.extend(t for t, _, _ in self._tokenize_pretoken(p))
        return out

    def encode_with_offsets(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        from motok.pretokenize import pretokenize_with_offsets
        toks, offs = [], []
        for p, o in pretokenize_with_offsets(text):
            for t, s0, e0 in self._tokenize_pretoken(p):
                toks.append(t); offs.append((o + s0, o + e0))
        return toks, offs

    def decode(self, tokens: list[str]) -> str:
        buf, out = bytearray(), []
        for t in tokens:
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                buf.append(int(t[3:5], 16))
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
                out.append(t)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)
