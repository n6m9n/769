#!/usr/bin/env python3
"""Trace Pareto fronts of the multi-objective vocabulary LP with epsilon-constraint sweeps.

  python scripts/04_sweep.py --config configs/smoke.yaml [--sweeps parity morph grid]

Sweeps (all warm-started: only column costs/bounds change between solves):
  parity : min Ttot  s.t. T_l <= tau for all l      tau from min-max optimum to the unconstrained worst language
  morph  : min Ttot  s.t. Mtot <= eps               eps from min-morph optimum to the unconstrained value
  grid   : min Ttot  s.t. T_l <= tau, Mtot <= eps   small 3x3 grid (3-objective front)
For each point we record LP values, the integer values after rounding + inference,
the integrality gap, and the reduced cost of the bounded aggregate = d(Ttot)/d(eps), the
local exchange rate between objectives (shadow price).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from motok.config import load_cfg
from motok.lp.model import VocabLP
from motok.toast.io import save_tokenizer
from motok.toast.problem import Problem

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--sweeps", nargs="+", default=["parity", "morph", "grid"])
ap.add_argument("--delta", type=float, default=1e-3, help="tie-break weight on Ttot for anchor solves")
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
cfg = load_cfg(a.config)
prob = Problem.load(cfg.out / "problem.pkl")
tokdir = cfg.out / "tokenizers"

t0 = time.time()
lp = VocabLP(prob, m=cfg.vocab_size, verbose=a.verbose)
print(f"[sweep] LP built in {lp.build_seconds:.1f}s: {lp.nX} token vars, {lp.nZ} node vars, {lp.nR} rows")
records = []


def run(tag, sweep, warm=True, **info):
    res = lp.solve(warm=warm)
    vocab = lp.round(res)
    ev = lp.evaluate(vocab)
    gap = (ev["Ttot"] - res.agg["Ttot"]) / max(ev["Ttot"], 1e-12)
    rec = {"tag": tag, "sweep": sweep, **info, "status": res.status, "lp": {k: float(v) for k, v in res.agg.items()},
           "int": {k: float(v) for k, v in ev.items()}, "gap_Ttot": float(gap),
           "rc": {k: float(v) for k, v in res.reduced_costs.items()},
           "n_frac_x": res.n_frac_x, "n_frac_z": res.n_frac_z, "seconds": res.seconds, "iters": res.iterations,
           "warm": bool(warm), "fallback": res.fallback, "warm_seconds": res.warm_seconds}
    records.append(rec)
    save_tokenizer(tokdir / f"toast_{tag}.pkl", vocab, prob.ngrams, prob.hierarchy,
                   meta={k: rec[k] for k in ["tag", "sweep", "lp", "int", "gap_Ttot"]},
                   morph_models=prob.meta.get("morph_models") if "morph" in prob.hierarchy else None)
    print(f"  {tag:<22} {res.status:<8} Ttot={ev['Ttot']:.2f} Tmax={ev['Tmax']:.2f} Mtot={ev['Mtot']:.2f} "
          f"gap={gap:.1e} frac(x,z)=({res.n_frac_x},{res.n_frac_z}) {res.seconds:.1f}s/{res.iterations}it"
          + (f" [warm start timed out after {res.warm_seconds:.0f}s -> cold]" if res.fallback else ""), flush=True)
    return res, ev


# ---- anchors (ideal points of each objective)
print("[anchors]")
lp.set_bounds(); lp.set_objective(Ttot=1.0)
r_comp, e_comp = run("compression", "anchor", warm=False)                       # vanilla ToaST (multilingual)
lp.set_objective(Ttot=a.delta, t=1.0)
r_mm, e_mm = run("minmax", "anchor", warm=False)                                 # best possible worst-language cost
lp.set_objective(Ttot=a.delta, Mtot=1.0)
r_mo, e_mo = run("min_morph", "anchor", warm=False)                              # fewest spurious boundaries

n = cfg.sweep["n_points"]
if "parity" in a.sweeps:
    print("[sweep parity] min Ttot s.t. T_l <= tau")
    lp.set_bounds(); lp.set_objective(Ttot=1.0); lp.solve(warm=False)   # basis of the compression anchor
    for i, tau in enumerate(np.linspace(r_mm.agg["t"], r_comp.agg["Tmax"], n + 2)[1:-1][::-1]):
        lp.set_bounds(t_max=tau)
        run(f"parity_{i}", "parity", tau=float(tau))
if "morph" in a.sweeps:
    print("[sweep morph] min Ttot s.t. Mtot <= eps")
    lp.set_bounds(); lp.set_objective(Ttot=1.0); lp.solve(warm=False)
    for i, eps in enumerate(np.linspace(r_mo.agg["Mtot"], r_comp.agg["Mtot"], n + 2)[1:-1][::-1]):
        lp.set_bounds(Mtot_max=eps)
        run(f"morph_{i}", "morph", eps=float(eps))
if "grid" in a.sweeps:
    print("[sweep grid] min Ttot s.t. T_l <= tau, Mtot <= eps")
    lp.set_bounds(); lp.set_objective(Ttot=1.0); lp.solve(warm=False)
    taus = np.linspace(r_mm.agg["t"], r_comp.agg["Tmax"], 5)[1:-1]
    epss = np.linspace(r_mo.agg["Mtot"], r_comp.agg["Mtot"], 5)[1:-1]
    for i, tau in enumerate(taus):
        for k, eps in enumerate(epss):
            lp.set_bounds(t_max=tau, Mtot_max=eps)
            run(f"grid_{i}{k}", "grid", tau=float(tau), eps=float(eps))

lp.set_bounds()
json.dump({"langs": prob.langs, "vocab_size": cfg.vocab_size, "records": records},
          open(cfg.out / "sweep.json", "w"), indent=1)
print(f"[sweep] {len(records)} solves in {time.time()-t0:.0f}s -> {cfg.out/'sweep.json'}")
