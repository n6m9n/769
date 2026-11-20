"""Intrinsic evaluation shared by every tokenizer (split-tree or HF).

A tokenizer only needs: .encode(text)->list[str], .encode_with_offsets(text)->(tokens, [(s,e)]), .vocab_size
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np


# ------------------------------------------------------------------ compression
def compression(tok, texts: list[str], kappa: float = 1.0) -> tuple[dict, Counter]:
    n_tok = n_chars = n_bytes = n_words = 0
    freq: Counter = Counter()
    for t in texts:
        ids = tok.encode(t)
        freq.update(ids)
        n_tok += len(ids); n_chars += len(t); n_bytes += len(t.encode("utf-8")); n_words += len(t.split())
    return {
        "tokens": n_tok,
        "chars_per_token": n_chars / max(n_tok, 1),
        "bytes_per_token": n_bytes / max(n_tok, 1),
        "fertility": n_tok / max(n_words, 1),
        # same unit as the LP's T_l: tokens per 1000 English-equivalent characters
        "T": 1000.0 * kappa * n_tok / max(n_chars, 1),
    }, freq


def renyi_efficiency(freq: Counter, vocab_size: int, alpha: float = 2.5) -> float:
    """Zouhar et al. (2023): H_alpha(p) / log|V|."""
    p = np.array(list(freq.values()), float)
    p /= p.sum()
    h = math.log(np.sum(p ** alpha)) / (1 - alpha)
    return h / math.log(vocab_size)


def gini(values) -> float:
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def premiums(tok, parallel: dict[str, list[str]], ref: str = "en") -> dict:
    """Token premium vs. reference language on line-aligned parallel text (e.g. FLORES)."""
    counts = {l: sum(len(tok.encode(t)) for t in lines) for l, lines in parallel.items()}
    return {l: counts[l] / counts[ref] for l in counts}


# ------------------------------------------------------------------ morphology
def morph_alignment(tok, gold: list[tuple[str, list[str]]], leading_space: bool = True) -> dict:
    """Boundary precision/recall/F1 against gold morpheme boundaries, plus MorphScore-style recall
    (Arnett & Bergen 2025: only words the tokenizer actually splits) and spurious boundaries/word
    (the quantity the LP's M objective penalises)."""
    tp = fp = fn = 0
    ms_scores, spurious = [], []
    pre = " " if leading_space else ""
    for word, segs in gold:
        B, pos = set(), 0
        for s in segs[:-1]:
            pos += len(s); B.add(pos)
        _, offs = tok.encode_with_offsets(pre + word)
        P = {s - len(pre) for s, _ in offs if len(pre) < s < len(pre) + len(word)}
        tp += len(P & B); fp += len(P - B); fn += len(B - P)
        spurious.append(len(P - B))
        if P and B:
            ms_scores.append(len(P & B) / len(B))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "morphscore": float(np.mean(ms_scores)) if ms_scores else float("nan"),
            "spurious_per_word": float(np.mean(spurious)), "n_words": len(gold)}


# ------------------------------------------------------------------ sharing
def sharing(used: dict[str, set]) -> dict:
    out = {}
    for a, b in combinations(sorted(used), 2):
        inter, uni = len(used[a] & used[b]), len(used[a] | used[b])
        out[f"{a}-{b}"] = inter / max(uni, 1)
    allsets = list(used.values())
    multi = Counter(t for s in allsets for t in s)
    out["frac_tokens_multilang"] = sum(1 for c in multi.values() if c > 1) / max(len(multi), 1)
    return out


# ------------------------------------------------------------------ everything
def evaluate(tok, eval_texts: dict[str, list[str]], gold: dict[str, list] | None = None,
             parallel: dict[str, list[str]] | None = None, kappa: dict[str, float] | None = None) -> dict:
    res: dict = {"per_lang": {}, "name": getattr(tok, "name", "?"), "vocab_size": tok.vocab_size}
    all_freq: Counter = Counter()
    used = {}
    for l, texts in eval_texts.items():
        c, f = compression(tok, texts, (kappa or {}).get(l, 1.0))
        c["renyi_eff"] = renyi_efficiency(f, tok.vocab_size)
        res["per_lang"][l] = c
        all_freq.update(f)
        used[l] = set(f)
    T = [v["T"] for v in res["per_lang"].values()]
    res["T_mean"], res["T_max"], res["T_gini"] = float(np.mean(T)), float(np.max(T)), gini(T)
    res["renyi_eff"] = renyi_efficiency(all_freq, tok.vocab_size)
    res["sharing"] = sharing(used)
    if gold:
        res["morph"] = {l: morph_alignment(tok, g) for l, g in gold.items()}
        res["morph_f1_mean"] = float(np.mean([m["f1"] for m in res["morph"].values()]))
        res["spurious_mean"] = float(np.mean([m["spurious_per_word"] for m in res["morph"].values()]))
    if parallel:
        counts = {l: sum(len(tok.encode(t)) for t in lines) for l, lines in parallel.items()}
        res["flores_abs"] = {l: counts[l] / max(len(parallel[l]), 1) for l in counts}   # tokens per sentence
        ref = "en" if "en" in counts else next(iter(counts))
        pr = {l: counts[l] / counts[ref] for l in counts}
        res["premium"] = pr
        res["premium_max"], res["premium_gini"] = max(pr.values()), gini(list(pr.values()))
    return res


# ------------------------------------------------------------------ Pareto tools
def nondominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (all objectives minimised)."""
    P = np.asarray(points, float)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = np.all(P <= P[i], 1) & np.any(P < P[i], 1)
            if dom.any():
                keep[i] = False
    return keep


def hypervolume(points: np.ndarray, ref: np.ndarray, n_samples: int = 200_000, seed: int = 0) -> float:
    """Dominated hypervolume (minimisation). Exact in 2-D, Monte Carlo otherwise."""
    P = np.asarray(points, float); ref = np.asarray(ref, float)
    P = P[np.all(P < ref, 1)]
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 2:
        P = P[np.argsort(P[:, 0])]
        hv, best_y = 0.0, ref[1]
        for x, y in P:
            if y < best_y:
                hv += (ref[0] - x) * (best_y - y); best_y = y
        return hv
    rng = np.random.default_rng(seed)
    lo = P.min(0)
    S = rng.uniform(lo, ref, size=(n_samples, P.shape[1]))
    dom = np.zeros(n_samples, bool)
    for p in P:
        dom |= np.all(S >= p, 1)
    return float(dom.mean() * np.prod(ref - lo))
