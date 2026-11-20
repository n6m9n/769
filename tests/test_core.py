"""Core correctness tests. Run: pytest -q"""
import itertools
from collections import Counter

import numpy as np
import pytest

from motok.eval.metrics import gini, hypervolume, nondominated
from motok.lp.model import VocabLP
from motok.pretokenize import pretokenize
from motok.toast.problem import build_problem
from motok.toast.trees import (SplitTreeTokenizer, annotate_morph, build_tree, make_levels, morph_boundaries,
                               ngram_counts, ngram_counts_naive, tree_tokenize)

TOY = {
    "en": Counter({" the": 50, " then": 20, " there": 15, " these": 12, " theme": 6, " other": 9, " there's": 3,
                   " walking": 8, " talking": 7, " walked": 6, " talked": 5}),
    "hu": Counter({" ház": 30, " házak": 12, " házakban": 8, " kert": 10, " kertek": 6, " kertekben": 4}),
}


def toy_problem(hierarchy=("grapheme", "char"), segs=None):
    return build_problem(TOY, trees_per_lang=100, hierarchy=list(hierarchy), segmenters=segs, verbose=False)


def test_pretokenizer_keeps_indic_marks():
    toks = pretokenize("नमस्ते दुनिया")
    assert toks == ["नमस्ते", " दुनिया"]


def test_tree_is_full_binary_with_char_leaves():
    prob = toy_problem()
    for t in prob.trees:
        L = len(t.text)
        assert t.n_nodes == 2 * L - 1
        assert all(t.end[k] - t.start[k] == 1 for k in t.leaves())
        for k in range(t.n_nodes):
            if t.left[k] >= 0:
                assert t.start[t.left[k]] == t.start[k] and t.end[t.right[k]] == t.end[k]
                assert t.end[t.left[k]] == t.start[t.right[k]]


def test_inference_covers_pretoken_exactly():
    prob = toy_problem()
    vocab = set(prob.base_chars) | {" the", "walk", "ing", " ház"}
    for t in prob.trees:
        toks = [t.token(k) for k in tree_tokenize(t, vocab)]
        assert "".join(toks) == t.text


def test_morph_first_trees_split_at_morph_boundaries():
    segs = {"hu": lambda w: {"házakban": ["ház", "ak", "ban"]}.get(w, [w]), "en": lambda w: [w]}
    prob = toy_problem(("morph", "grapheme", "char"), segs)
    t = next(t for t in prob.trees if t.text == " házakban")
    # root's first split must be at a morph boundary (positions 1, 4, 6)
    assert t.end[t.left[0]] in {1, 4, 6}
    assert t.misaligned.sum() > 0 and t.aligned.sum() > 0


def _brute_force_opt(lp: VocabLP, m_free: int):
    """Enumerate all vocabularies (tiny problems only) and return the min total tokens."""
    nb = len(lp.prob.base_chars)
    cand = lp.tokens[nb:]
    best = np.inf
    for combo in itertools.combinations(cand, m_free):
        v = list(lp.prob.base_chars) + list(combo)
        best = min(best, lp.evaluate(v)["Ttot"])
    return best


def test_lp_matches_brute_force_on_toy():
    small = {"en": Counter({" the": 10, " then": 4, " other": 3})}
    prob = build_problem(small, trees_per_lang=10, verbose=False)
    nb = len(prob.base_chars)
    for m_free in [1, 2, 3]:
        lp = VocabLP(prob, m=nb + m_free + 256)
        lp.set_objective(Ttot=1.0)
        res = lp.solve()
        vocab = lp.round(res)
        ev = lp.evaluate(vocab)
        bf = _brute_force_opt(lp, m_free)
        assert res.agg["Ttot"] <= bf + 1e-6          # LP is a valid lower bound
        assert ev["Ttot"] == pytest.approx(bf, rel=1e-9) or ev["Ttot"] >= bf  # rounded is feasible
        assert ev["Ttot"] <= bf * 1.05               # and near-optimal on this toy


def test_integral_lp_equals_inference():
    prob = toy_problem()
    lp = VocabLP(prob, m=len(prob.base_chars) + 8 + 256)
    lp.set_objective(Ttot=1.0)
    res = lp.solve()
    ev = lp.evaluate(lp.round(res))
    if res.n_frac_x == 0:
        assert ev["Ttot"] == pytest.approx(res.agg["Ttot"], rel=1e-9)
    assert ev["Ttot"] >= res.agg["Ttot"] - 1e-9


def test_minmax_reduces_worst_language():
    prob = toy_problem()
    lp = VocabLP(prob, m=len(prob.base_chars) + 6 + 256)
    lp.set_objective(Ttot=1.0); r0 = lp.solve(warm=False)
    lp.set_objective(Ttot=1e-3, t=1.0); r1 = lp.solve(warm=False)
    assert r1.agg["t"] <= r0.agg["Tmax"] + 1e-9
    assert r1.agg["Ttot"] >= r0.agg["Ttot"] - 1e-9


def test_warm_time_limit_is_per_call():
    """HiGHS's time_limit is cumulative over run() calls; the warm budget must not count earlier solves."""
    prob = toy_problem()
    lp = VocabLP(prob, m=len(prob.base_chars) + 6 + 256)
    lp.set_objective(Ttot=1.0)
    while lp.h.getRunTime() < 0.05:          # accumulate run-clock time with cold solves
        r0 = lp.solve(warm=False)
    # a feasible bound strictly between the min-max optimum and the anchor's worst language
    # (if they coincide - the anchor is already min-max optimal - just restate the anchor's value)
    lp.set_objective(t=1.0); t_mm = lp.solve(warm=False).agg["t"]
    lp.set_objective(Ttot=1.0); r0 = lp.solve(warm=False)
    tau = 0.5 * (t_mm + r0.agg["Tmax"]) if r0.agg["Tmax"] > t_mm + 1e-6 else r0.agg["Tmax"] + 1e-6
    budget = lp.h.getRunTime() / 2           # < time already used, >> a warm re-solve of the toy
    lp.set_bounds(t_max=tau)
    r1 = lp.solve(warm=True, warm_time_limit=budget)
    assert not r1.fallback and r1.status == "Optimal"


def test_roundtrip_with_byte_fallback():
    prob = toy_problem()
    tok = SplitTreeTokenizer(list(prob.base_chars) + [" the", "ing"], prob.ngrams, prob.hierarchy)
    for s in ["the walking dog", "日本語 ok", "házakban 123!"]:
        assert tok.decode(tok.encode(s)) == s
        toks, offs = tok.encode_with_offsets(s)
        assert len(toks) == len(offs)


def test_pareto_utils():
    P = np.array([[1, 5], [2, 3], [3, 4], [4, 1]])
    assert list(nondominated(P)) == [True, True, False, True]
    assert hypervolume(P[nondominated(P)], np.array([5, 6])) == pytest.approx(4*1 + 3*2 + 1*2)
    assert gini([1, 1, 1]) == pytest.approx(0) and gini([0, 0, 1]) > 0.5


def test_apriori_ngrams_match_naive():
    merged = Counter()
    for c in TOY.values():
        merged.update(c)
    merged.update({"x" * 70: 2, " अनुवाद": 4, " ab": 1})
    for mc in [1, 3, 5, 12, 40]:
        assert ngram_counts(dict(merged), mc, max_len=64) == ngram_counts_naive(dict(merged), mc, max_len=64)


def test_parallel_counting_matches_serial(tmp_path):
    from motok.pretokenize import count_pretokens_file
    f = tmp_path / "t.txt"
    f.write_text("\n".join(["the walking dog házakban", "नमस्ते दुनिया 123", "there's  more!"] * 50) + "\n")
    assert count_pretokens_file(f, n_proc=1) == count_pretokens_file(f, n_proc=3, chunk_lines=7)
