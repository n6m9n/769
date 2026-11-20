#!/usr/bin/env python3
"""Trace Pareto fronts of the multi-objective vocabulary LP.

  python scripts/04_sweep.py --config configs/smoke.yaml [--sweeps parity morph grid wsum] [--out-name NAME]

epsilon-constraint sweeps (warm-started dual simplex: only column bounds change between solves)
  parity : min Ttot  s.t. T_l <= tau for all l      tau from min-max optimum to the unconstrained worst language
  morph  : min Ttot  s.t. Mtot <= eps               eps from min-morph optimum to the unconstrained value
  grid   : min Ttot  s.t. T_l <= tau, Mtot <= eps   eps chosen *per tau* in [min Mtot | tau, Mtot at min Ttot | tau]
weighted-sum sweeps (warm-started primal simplex: only column costs change)
  wsum   : min Ttot + lam * t   and   min Ttot + mu * Mtot   for geometric lam, mu
For each point we record LP values, the integer values after rounding + inference, the integrality gap,
and the reduced cost of the bounded aggregate = d(Ttot)/d(bound), the local exchange rate (shadow price).

Bound shading (fixes rounding overshoot): if the rounded vocabulary violates the point's epsilon-bound,
the bound is tightened by the overshoot, the LP re-solved (warm) and re-rounded, up to --max-shade times.
The LP value at the *original* bound stays the lower bound, so gap_Ttot is a valid certificate whenever
the final integer point satisfies the original bound (recorded as int_feasible).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from motok.config import ROOT, load_cfg
from motok.lp.model import VocabLP
from motok.toast.io import save_tokenizer
from motok.toast.problem import Problem

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--sweeps", nargs="+", default=["parity", "morph", "grid", "wsum"])
ap.add_argument("--out-name", default=None, help="results/<out-name> instead of the config results_dir")
ap.add_argument("--n-points", type=int, default=None)
ap.add_argument("--vocab-size", type=int, default=None)
ap.add_argument("--delta", type=float, default=1e-3, help="tie-break weight on Ttot for anchor solves")
ap.add_argument("--max-shade", type=int, default=4)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--no-save", action="store_true", help="don't save tokenizers (timing / scaling runs)")
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
cfg = load_cfg(a.config)
if a.out_name:
    cfg["results_dir"] = f"results/{a.out_name}"
if a.vocab_size:
    cfg["vocab_size"] = a.vocab_size
prob = Problem.load(cfg.out / "problem.pkl")
tokdir = cfg.out / "tokenizers"

t0 = time.time()
lp = VocabLP(prob, m=cfg.vocab_size, threads=a.threads, verbose=a.verbose)
print(f"[sweep] LP built in {lp.build_seconds:.1f}s: {lp.nX} token vars, {lp.nZ} node vars, {lp.nR} rows", flush=True)
records = []
AGG = {"t_max": "Tmax", "Mtot_max": "Mtot"}
TOL = 1e-6


def overshoot(ev, bounds):
    return {k: ev[AGG[k]] - b for k, b in bounds.items() if ev[AGG[k]] > b * (1 + TOL) + TOL}


def run(tag, sweep, warm=True, method="dual", bounds=None, **info):
    bounds = bounds or {}
    res = lp.solve(warm=warm, method=method)
    base = {"tag": tag, "sweep": sweep, **info, "status": res.status, "seconds": res.seconds, "iters": res.iterations,
            "warm": bool(warm), "fallback": res.fallback, "warm_seconds": res.warm_seconds}
    if res.status != "Optimal":   # e.g. a grid corner where tau and eps can't hold together: no Pareto point
        records.append(base)
        print(f"  {tag:<22} {res.status:<8} (no tokenizer saved) {res.seconds:.1f}s/{res.iterations}it", flush=True)
        return res, None
    vocab = lp.round(res)
    ev = lp.evaluate(vocab)
    shade, shade_sec, cur = 0, 0.0, dict(bounds)
    while bounds and overshoot(ev, bounds) and shade < a.max_shade:
        over = overshoot(ev, bounds)
        for k, d in over.items():
            cur[k] = cur[k] - 1.25 * d - 1e-4 * bounds[k]
        lp.set_bounds(**cur)
        r2 = lp.solve(warm=True, method="dual")
        shade += 1; shade_sec += r2.seconds
        if r2.status != "Optimal":
            break
        vocab = lp.round(r2)
        ev = lp.evaluate(vocab)
    if shade:
        lp.set_bounds(**bounds)
    gap = (ev["Ttot"] - res.agg["Ttot"]) / max(ev["Ttot"], 1e-12)
    rec = {**base, "lp": {k: float(v) for k, v in res.agg.items()}, "int": {k: float(v) for k, v in ev.items()},
           "gap_Ttot": float(gap), "rc": {k: float(v) for k, v in res.reduced_costs.items()},
           "n_frac_x": res.n_frac_x, "n_frac_z": res.n_frac_z, "shade_steps": shade, "shade_seconds": shade_sec,
           "int_feasible": not overshoot(ev, bounds)}
    records.append(rec)
    if not a.no_save:
        save_tokenizer(tokdir / f"toast_{tag}.pkl", vocab, prob.ngrams, prob.hierarchy,
                       meta={k: rec.get(k) for k in ["tag", "sweep", "lp", "int", "gap_Ttot", "tau", "eps", "lam", "mu"]},
                       morph_models=prob.meta.get("morph_models") if "morph" in prob.hierarchy else None)
    print(f"  {tag:<22} {res.status:<8} Ttot={ev['Ttot']:.2f} Tmax={ev['Tmax']:.2f} Mtot={ev['Mtot']:.2f} "
          f"gap={gap:.1e} frac(x,z)=({res.n_frac_x},{res.n_frac_z}) {res.seconds:.1f}s/{res.iterations}it"
          + (f" shade={shade}{'' if rec['int_feasible'] else ' (still violates)'}" if shade else "")
          + (f" [warm start timed out after {res.warm_seconds:.0f}s -> cold]" if res.fallback else ""), flush=True)
    return res, ev


def save():
    lp.set_bounds()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "?"
    json.dump({"langs": prob.langs, "vocab_size": cfg.vocab_size, "build_seconds": lp.build_seconds,
               "n_tokens": lp.nX, "n_nodes": lp.nZ, "n_rows": lp.nR, "n_trees": len(prob.trees),
               "trees_per_lang": prob.meta.get("trees_per_lang"), "coverage": prob.coverage,
               "commit": commit, "total_seconds": time.time() - t0, "records": records},
              open(cfg.out / "sweep.json", "w"), indent=1)


# ---- anchors (ideal points of each objective)
print("[anchors]", flush=True)
lp.set_bounds(); lp.set_objective(Ttot=1.0)
r_comp, e_comp = run("compression", "anchor", warm=False)                       # vanilla ToaST (multilingual)
lp.set_objective(Ttot=a.delta, t=1.0)
r_mm, e_mm = run("minmax", "anchor", warm=False)                                 # best possible worst-language cost
lp.set_objective(Ttot=a.delta, Mtot=1.0)
r_mo, e_mo = run("min_morph", "anchor", warm=False)                              # fewest spurious boundaries
save()

n = a.n_points or cfg.sweep["n_points"]
if "parity" in a.sweeps:
    print("[sweep parity] min Ttot s.t. T_l <= tau", flush=True)
    lp.set_bounds(); lp.set_objective(Ttot=1.0); lp.solve(warm=False)   # basis of the compression anchor
    for i, tau in enumerate(np.linspace(r_mm.agg["t"], r_comp.agg["Tmax"], n + 2)[1:-1][::-1]):
        lp.set_bounds(t_max=tau)
        run(f"parity_{i}", "parity", bounds={"t_max": float(tau)}, tau=float(tau))
    save()
if "morph" in a.sweeps:
    print("[sweep morph] min Ttot s.t. Mtot <= eps", flush=True)
    lp.set_bounds(); lp.set_objective(Ttot=1.0); lp.solve(warm=False)
    for i, eps in enumerate(np.linspace(r_mo.agg["Mtot"], r_comp.agg["Mtot"], n + 2)[1:-1][::-1]):
        lp.set_bounds(Mtot_max=eps)
        run(f"morph_{i}", "morph", bounds={"Mtot_max": float(eps)}, eps=float(eps))
    save()
if "grid" in a.sweeps:
    print("[sweep grid] min Ttot s.t. T_l <= tau, Mtot <= eps (eps range per tau)", flush=True)
    taus = np.linspace(r_mm.agg["t"], r_comp.agg["Tmax"], 5)[1:-1][::-1]     # loose -> tight
    for i, tau in enumerate(taus):
        lp.set_bounds(t_max=tau)
        lp.set_objective(Ttot=a.delta, Mtot=1.0); lo = lp.solve(warm=False).agg["Mtot"]   # min Mtot | tau
        lp.set_objective(Ttot=1.0); hi = lp.solve(warm=False).agg["Mtot"]                 # Mtot at min Ttot | tau
        for k, eps in enumerate(np.linspace(lo, hi, 5)[1:-1][::-1]):
            lp.set_bounds(t_max=tau, Mtot_max=eps)
            run(f"grid_{i}{k}", "grid", bounds={"t_max": float(tau), "Mtot_max": float(eps)},
                tau=float(tau), eps=float(eps), eps_range=[float(lo), float(hi)])
    save()
if "wsum" in a.sweeps:
    print("[sweep wsum] min Ttot + lam*t  /  min Ttot + mu*Mtot (warm primal)", flush=True)
    lp.set_bounds()
    for key, name, grid in [("t", "wpar", np.geomspace(0.02, 5.0, n)), ("Mtot", "wmor", np.geomspace(0.05, 20.0, n))]:
        lp.set_objective(Ttot=1.0); lp.solve(warm=False)
        for i, w in enumerate(grid):
            lp.set_objective(Ttot=1.0, **{key: float(w)})
            run(f"{name}_{i}", name, method="primal", **({"lam": float(w)} if key == "t" else {"mu": float(w)}))
    save()

save()
print(f"[sweep] {len(records)} solves in {time.time()-t0:.0f}s -> {cfg.out/'sweep.json'}", flush=True)
