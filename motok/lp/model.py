"""Multi-objective vocabulary-selection LP over split trees ("Pareto-ToaST").

Variables
  x_i  in [0,1]   token i in vocabulary                     (base chars fixed to 1)
  z_k  in [0,1]   node k (of some tree j) is emitted
Core constraints (ToaST):
  sum_i x_i = m                                            vocabulary size
  sum_{k on leaf->root path} z_k = 1   for every leaf      every char covered exactly once
  z_k <= x_{tok(k)}                                         only vocabulary tokens are used
Aggregate columns (all linear in z / x):
  T_l   = 1000 * kappa_l / chars_l * sum_k c_{j(k),l} z_k          tokens per 1000 (English-equiv.) chars
  M_l   = 1000 * kappa_l / chars_l * sum_k c_{j(k),l} mis_k z_k    spurious (inside-morpheme) boundaries
  A_l   = same with aligned_k                                       correct morpheme boundaries hit
  Ttot  = sum_l pi_l T_l,  Mtot = sum_l pi_l M_l   (pi_l = data share of chars)
  t    >= T_l  for all l                                            worst-language cost (min-max parity)
  S     = sum_i s_i x_i                                             cross-lingual sharing score
Scalarisations are just costs / bounds on aggregate columns, so every sweep warm-starts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import highspy
import numpy as np
import scipy.sparse as sp

from motok.toast.problem import Problem
from motok.toast.trees import tree_tokenize

INF = highspy.kHighsInf


@dataclass
class LPResult:
    status: str
    objective: float
    agg: dict                 # aggregate values from the LP (fractional) solution
    reduced_costs: dict       # reduced costs of aggregate columns (shadow prices of their bounds)
    def_duals: dict           # duals of the aggregate-definition rows
    seconds: float
    iterations: int
    n_frac_x: int
    n_frac_z: int
    x: np.ndarray = field(repr=False, default=None)
    z: np.ndarray = field(repr=False, default=None)


class VocabLP:
    def __init__(self, prob: Problem, m: int, reserved: int = 256, with_sharing: bool = True,
                 threads: int = 0, verbose: bool = False):
        t0 = time.time()
        self.prob, self.m, self.reserved = prob, m, reserved
        langs, trees, C = prob.langs, prob.trees, prob.counts
        L = len(langs)

        # ---- candidate tokens and node -> token map
        tok_index: dict[str, int] = {ch: i for i, ch in enumerate(prob.base_chars)}
        node_tok, node_tree, node_mis, node_ali = [], [], [], []
        node_off = np.zeros(len(trees) + 1, dtype=np.int64)
        for j, t in enumerate(trees):
            node_off[j + 1] = node_off[j] + t.n_nodes
            for k in range(t.n_nodes):
                s = t.token(k)
                i = tok_index.get(s)
                if i is None:
                    i = tok_index[s] = len(tok_index)
                node_tok.append(i)
            node_tree.append(np.full(t.n_nodes, j))
            node_mis.append(t.misaligned if t.misaligned is not None else np.zeros(t.n_nodes))
            node_ali.append(t.aligned if t.aligned is not None else np.zeros(t.n_nodes))
        self.tokens = list(tok_index)
        self.node_off = node_off
        node_tok = np.asarray(node_tok)
        node_tree = np.concatenate(node_tree)
        node_mis = np.concatenate(node_mis).astype(float)
        node_ali = np.concatenate(node_ali).astype(float)
        self.node_tok, self.node_tree = node_tok, node_tree
        nX, nZ = len(self.tokens), len(node_tok)
        self.nX, self.nZ = nX, nZ

        # ---- per-language coefficients
        scale = 1000.0 * prob.kappa / np.maximum(prob.chars, 1)          # [L]
        self.coef_T = C[node_tree] * scale[None, :]                       # [nZ, L]
        self.coef_M = self.coef_T * node_mis[:, None]
        self.coef_A = self.coef_T * node_ali[:, None]
        self.pi = prob.chars / prob.chars.sum()

        # sharing score s_i = (#languages whose trees contain token i) - 1
        langs_per_tok = np.zeros((nX, L), dtype=bool)
        present = C[node_tree] > 0
        for l in range(L):
            langs_per_tok[node_tok[present[:, l]], l] = True
        self.share = np.maximum(langs_per_tok.sum(1) - 1, 0).astype(float)

        # ---- column layout
        self.cX = 0
        self.cZ = nX
        a = nX + nZ
        self.cT = np.arange(a, a + L); a += L
        self.cM = np.arange(a, a + L); a += L
        self.cA = np.arange(a, a + L); a += L
        self.cTtot, self.cMtot, self.ct, self.cS = a, a + 1, a + 2, a + 3
        nC = a + 4
        self.nC = nC

        rows, cols, vals, lo, hi = [], [], [], [], []
        r = 0

        def add_row(cidx, cval, lb, ub):
            nonlocal r
            rows.append(np.full(len(cidx), r)); cols.append(np.asarray(cidx)); vals.append(np.asarray(cval, float))
            lo.append(lb); hi.append(ub); r += 1

        # vocab size
        self.r_vocab = r
        add_row(np.arange(nX), np.ones(nX), m - reserved, m - reserved)
        # leaf coverage: one row per leaf, entries = path nodes
        for j, t in enumerate(trees):
            base = node_off[j]
            for leaf in t.leaves():
                path = t.path_to_root(int(leaf))
                add_row(base + np.asarray(path) + self.cZ, np.ones(len(path)), 1.0, 1.0)
        # link z_k - x_tok(k) <= 0   (vectorised)
        r0 = r
        rows.append(np.repeat(np.arange(r0, r0 + nZ), 2))
        cols.append(np.stack([self.cZ + np.arange(nZ), node_tok], 1).ravel())
        vals.append(np.tile([1.0, -1.0], nZ))
        lo.extend([-INF] * nZ); hi.extend([0.0] * nZ); r += nZ
        # aggregate definitions
        self.r_defT, self.r_defM, self.r_defA = [], [], []
        for l in range(L):
            for coef, col, store in [(self.coef_T, self.cT, self.r_defT), (self.coef_M, self.cM, self.r_defM),
                                     (self.coef_A, self.cA, self.r_defA)]:
                nz = np.nonzero(coef[:, l])[0]
                store.append(r)
                add_row(np.concatenate([[col[l]], self.cZ + nz]), np.concatenate([[1.0], -coef[nz, l]]), 0.0, 0.0)
        self.r_defTtot = r
        add_row(np.concatenate([[self.cTtot], self.cT]), np.concatenate([[1.0], -self.pi]), 0.0, 0.0)
        self.r_defMtot = r
        add_row(np.concatenate([[self.cMtot], self.cM]), np.concatenate([[1.0], -self.pi]), 0.0, 0.0)
        for l in range(L):  # T_l - t <= 0
            add_row([self.cT[l], self.ct], [1.0, -1.0], -INF, 0.0)
        if with_sharing:
            nzs = np.nonzero(self.share)[0]
            add_row(np.concatenate([[self.cS], nzs]), np.concatenate([[1.0], -self.share[nzs]]), 0.0, 0.0)
        self.nR = r

        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nC))
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nC, r
        col_lo = np.zeros(nC); col_hi = np.ones(nC)
        col_lo[: len(prob.base_chars)] = 1.0               # base chars always in vocabulary
        agg = np.r_[self.cT, self.cM, self.cA, [self.cTtot, self.cMtot, self.ct, self.cS]]
        col_lo[agg] = 0.0; col_hi[agg] = INF
        col_lo[self.cS] = -INF
        lp.col_lower_, lp.col_upper_ = col_lo, col_hi
        lp.col_cost_ = np.zeros(nC)
        lp.row_lower_, lp.row_upper_ = np.array(lo), np.array(hi)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", verbose)
        self.h.setOptionValue("solver", "simplex")
        self.h.setOptionValue("simplex_strategy", 1)  # dual simplex: warm starts after bound/RHS changes
        if threads:
            self.h.setOptionValue("threads", threads)
        self.h.passModel(lp)
        self._col_hi = col_hi.copy()
        self.build_seconds = time.time() - t0
        if len(prob.base_chars) > m - reserved:
            raise ValueError(f"vocab {m} too small: {len(prob.base_chars)} base chars + {reserved} reserved bytes")
        if verbose:
            print(f"[lp] {nC} cols, {r} rows, {A.nnz} nnz, built in {self.build_seconds:.1f}s")

    # ------------------------------------------------------------- scalarisation
    def set_objective(self, Ttot=0.0, T=None, Mtot=0.0, M=None, A=None, t=0.0, S=0.0):
        """Linear costs on aggregate columns (minimisation).
        T, M: per-language costs (scalar or array). A, S: *rewards* (subtracted)."""
        L = len(self.prob.langs)
        vec = lambda v: np.zeros(L) if v is None else np.broadcast_to(np.asarray(v, float), (L,))
        for c, v in [(self.cTtot, Ttot), (self.cMtot, Mtot), (self.ct, t), (self.cS, -S)]:
            self.h.changeColCost(int(c), float(v))
        for cols, v, sign in [(self.cT, vec(T), 1.0), (self.cM, vec(M), 1.0), (self.cA, vec(A), -1.0)]:
            for l in range(L):
                self.h.changeColCost(int(cols[l]), sign * float(v[l]))

    def set_bounds(self, Mtot_max=None, M_max=None, T_max=None, t_max=None):
        """epsilon-constraints: upper bounds on aggregates (None = unbounded)."""
        L = len(self.prob.langs)
        self.h.changeColBounds(int(self.cMtot), 0.0, INF if Mtot_max is None else float(Mtot_max))
        self.h.changeColBounds(int(self.ct), 0.0, INF if t_max is None else float(t_max))
        for arr, cols in [(M_max, self.cM), (T_max, self.cT)]:
            v = [None] * L if arr is None else list(np.broadcast_to(np.asarray(arr, float), (L,)))
            for l in range(L):
                self.h.changeColBounds(int(cols[l]), 0.0, INF if v[l] is None else float(v[l]))

    def set_vocab_size(self, m):
        self.m = m
        self.h.changeRowBounds(self.r_vocab, m - self.reserved, m - self.reserved)

    # ------------------------------------------------------------- solve
    def solve(self, warm: bool = True, method: str = "dual", eps=1e-5, warm_time_limit: float | None = 60.0) -> LPResult:
        """warm=False discards the basis so HiGHS runs presolve (best after objective changes).
        warm=True re-optimises from the previous basis: use method="dual" after bound/RHS changes
        (epsilon-constraint sweeps, vocab size) - the old basis stays dual feasible - and
        method="primal" after small cost changes (weighted-sum sweeps)."""
        t0 = time.time()
        self.h.setOptionValue("simplex_strategy", 1 if method == "dual" else 4)
        cold = not warm or not getattr(self, "_solved_once", False)
        if cold:
            self.h.clearSolver()
        self.h.setOptionValue("time_limit", float(warm_time_limit) if (not cold and warm_time_limit) else INF)
        self.h.run()
        if not cold and self.h.getModelStatus() == highspy.HighsModelStatus.kTimeLimit:
            # warm start stalled (degenerate re-optimisation, seen in the morph sweep): fall back to presolve
            self.h.clearSolver()
            self.h.setOptionValue("time_limit", INF)
            self.h.run()
        self._solved_once = True
        status = self.h.modelStatusToString(self.h.getModelStatus())
        sol, info = self.h.getSolution(), self.h.getInfo()
        cv = np.asarray(sol.col_value); cd = np.asarray(sol.col_dual); rd = np.asarray(sol.row_dual)
        x, z = cv[: self.nX], cv[self.cZ: self.cZ + self.nZ]
        langs = self.prob.langs
        agg = {f"T_{l}": cv[self.cT[i]] for i, l in enumerate(langs)}
        agg.update({f"M_{l}": cv[self.cM[i]] for i, l in enumerate(langs)})
        agg.update({f"A_{l}": cv[self.cA[i]] for i, l in enumerate(langs)})
        agg.update(Ttot=cv[self.cTtot], Mtot=cv[self.cMtot], t=cv[self.ct], S=cv[self.cS],
                   Tmax=max(cv[self.cT]))
        rc = {"Mtot": cd[self.cMtot], "t": cd[self.ct], **{f"T_{l}": cd[self.cT[i]] for i, l in enumerate(langs)},
              **{f"M_{l}": cd[self.cM[i]] for i, l in enumerate(langs)}}
        dd = {"T": rd[self.r_defT], "M": rd[self.r_defM], "A": rd[self.r_defA]}
        fx = int(np.sum((x > eps) & (x < 1 - eps)))
        fz = int(np.sum((z > eps) & (z < 1 - eps)))
        return LPResult(status, info.objective_function_value, agg, rc, dd, time.time() - t0,
                        int(info.simplex_iteration_count), fx, fz, x.copy(), z.copy())

    # ------------------------------------------------------------- rounding
    def round(self, res: LPResult, eps=1e-5) -> list[str]:
        """ToaST rounding: keep x~1, fill remaining slots with fractional tokens of largest
        objective contribution. Node weights use the LP's own marginal values (|row duals|
        of the aggregate definitions), so the rule works for any scalarisation."""
        x, z = res.x, res.z
        target = self.m - self.reserved
        I1 = np.nonzero(x >= 1 - eps)[0]
        IF = np.nonzero((x > eps) & (x < 1 - eps))[0]
        gT, gM, gA = np.abs(res.def_duals["T"]), np.abs(res.def_duals["M"]), np.abs(res.def_duals["A"])
        w = self.coef_T @ gT + self.coef_M @ gM + self.coef_A @ gA
        if not np.any(w > 0):
            w = self.coef_T.sum(1)
        contrib = np.bincount(self.node_tok, weights=z * w, minlength=self.nX)
        nb = len(self.prob.base_chars)
        if len(I1) > target:  # numerical edge case: trim lowest contribution (never base chars)
            keep_nb = I1[I1 >= nb]
            I1 = np.r_[np.arange(nb), keep_nb[np.argsort(-contrib[keep_nb])][: target - nb]]
        need = target - len(I1)
        chosen = IF[np.argsort(-contrib[IF])][:need] if need > 0 else np.array([], int)
        vocab_idx = np.union1d(I1, chosen)
        if len(vocab_idx) < target:  # LP put mass on exactly-zero tokens? fill by contribution
            rest = np.setdiff1d(np.arange(self.nX), vocab_idx)
            vocab_idx = np.union1d(vocab_idx, rest[np.argsort(-contrib[rest])][: target - len(vocab_idx)])
        return [self.tokens[i] for i in vocab_idx]

    # ------------------------------------------------------------- exact integer evaluation
    def evaluate(self, vocab: list[str]) -> dict:
        """Run split-tree inference with an integer vocabulary and compute the same aggregates."""
        vset = set(vocab)
        z = np.zeros(self.nZ)
        for j, t in enumerate(self.prob.trees):
            for k in tree_tokenize(t, vset):
                z[self.node_off[j] + k] = 1.0
        T, M, A = self.coef_T.T @ z, self.coef_M.T @ z, self.coef_A.T @ z
        langs = self.prob.langs
        out = {f"T_{l}": T[i] for i, l in enumerate(langs)}
        out.update({f"M_{l}": M[i] for i, l in enumerate(langs)})
        out.update({f"A_{l}": A[i] for i, l in enumerate(langs)})
        tok_idx = {s: i for i, s in enumerate(self.tokens)}
        S = sum(self.share[tok_idx[s]] for s in vocab if s in tok_idx)
        out.update(Ttot=float(self.pi @ T), Mtot=float(self.pi @ M), t=float(T.max()), Tmax=float(T.max()), S=float(S))
        return out
