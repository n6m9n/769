#!/usr/bin/env python3
"""Sanity checks on a finished run (sweep.json + tokenizers + eval.json).

  python scripts/check_results.py --config configs/smoke.yaml

1. LP <= integer: for every sweep point, the scalarised objective of the LP solution must not exceed
   that of the rounded vocabulary. This is only a theorem if the rounded vocabulary also satisfies the
   point's epsilon-bounds, so bound violations by the integer point are reported next to it.
2. Fronts monotone: along each 1-D sweep (bound tightened), LP Ttot must be non-decreasing and the
   bounded aggregate must respect its bound (LP theory). Integer (training-side) and held-out values are
   checked too, but there non-monotonicity is possible (rounding, generalisation) and only flagged.
3. Round trip: decode(encode(s)) == s on *every* held-out line of every language, plus strings with
   characters unseen in training (byte fallback).
Exits non-zero if a hard check fails. Writes checks.json next to sweep.json.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from motok.tokload import load_any
from motok.config import load_cfg, read_lines

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--tol", type=float, default=1e-6, help="relative tolerance")
ap.add_argument("--max-lines", type=int, default=5000, help="held-out lines per language for the round trip")
a = ap.parse_args()
cfg = load_cfg(a.config)
S = json.load(open(cfg.out / "sweep.json"))
recs = S["records"]
DELTA = 1e-3  # 04_sweep.py default tie-break weight on Ttot for anchors
hard_fail, flags, out = [], [], {}


def objective(r, v):
    """Scalarised objective value of a record evaluated on aggregate dict v."""
    if r["tag"] == "minmax":
        return DELTA * v["Ttot"] + v["Tmax"]
    if r["tag"] == "min_morph":
        return DELTA * v["Ttot"] + v["Mtot"]
    return v["Ttot"]


def violations(r, v):
    out = {}
    if "tau" in r and v["Tmax"] > r["tau"] * (1 + a.tol):
        out["Tmax>tau"] = v["Tmax"] - r["tau"]
    if "eps" in r and v["Mtot"] > r["eps"] * (1 + a.tol):
        out["Mtot>eps"] = v["Mtot"] - r["eps"]
    return out


# ---- 1. LP <= integer
rows = []
infeasible = [r["tag"] for r in recs if r["status"] == "Infeasible"]
if infeasible:
    flags.append(f"infeasible epsilon-combinations (no point, no tokenizer): {infeasible}")
for r in recs:
    if r["status"] == "Infeasible":
        continue
    if "lp" not in r:
        hard_fail.append(f"{r['tag']}: LP status {r['status']} (no solution recorded)")
        continue
    lp, it = objective(r, r["lp"]), objective(r, r["int"])
    viol_lp, viol_int = violations(r, r["lp"]), violations(r, r["int"])
    ok = lp <= it * (1 + a.tol) + 1e-9
    rows.append({"tag": r["tag"], "lp_obj": lp, "int_obj": it, "rel_gap": (it - lp) / max(abs(it), 1e-12),
                 "lp_le_int": ok, "lp_bound_viol": viol_lp, "int_bound_viol": viol_int,
                 "status": r["status"], "seconds": r["seconds"], "fallback": r.get("fallback")})
    if r["status"] != "Optimal":
        hard_fail.append(f"{r['tag']}: LP status {r['status']}")
    if viol_lp:
        hard_fail.append(f"{r['tag']}: LP solution violates its own bound {viol_lp}")
    if not ok:
        (flags if viol_int else hard_fail).append(
            f"{r['tag']}: LP obj {lp:.6f} > integer obj {it:.6f}"
            + (f" (integer point infeasible for this point: {viol_int}, so not a contradiction)" if viol_int else ""))
    elif viol_int:
        flags.append(f"{r['tag']}: rounded vocab violates the epsilon-bound {viol_int}")
out["lp_vs_int"] = rows

# ---- 2. monotone fronts (training side)
mono = {}
for sw, bkey, agg in [("parity", "tau", "Tmax"), ("morph", "eps", "Mtot")]:
    R = sorted([r for r in recs if r["sweep"] == sw and "lp" in r], key=lambda r: -r[bkey])   # loose -> tight
    if not R:
        continue
    lpT = [r["lp"]["Ttot"] for r in R]; inT = [r["int"]["Ttot"] for r in R]
    inA = [r["int"][agg] for r in R]
    d = {"bounds": [r[bkey] for r in R], "lp_Ttot": lpT, "int_Ttot": inT, f"int_{agg}": inA}
    d["lp_monotone"] = bool(np.all(np.diff(lpT) >= -a.tol * max(lpT)))
    d["int_Ttot_monotone"] = bool(np.all(np.diff(inT) >= -a.tol * max(inT)))
    d[f"int_{agg}_monotone"] = bool(np.all(np.diff(inA) <= a.tol * max(inA)))
    mono[sw] = d
    if not d["lp_monotone"]:
        hard_fail.append(f"{sw}: LP Ttot not non-decreasing as the bound tightens: {np.round(lpT, 3)}")
    for k in ["int_Ttot_monotone", f"int_{agg}_monotone"]:
        if not d[k]:
            flags.append(f"{sw}: {k} is False (integer, training side): Ttot={np.round(inT, 2)}, {agg}={np.round(inA, 2)}")
out["monotone_train"] = mono

# ---- 2b. held-out fronts (from eval.json): sweep order, flag non-monotone steps
E = json.load(open(cfg.out / "eval.json")) if (cfg.out / "eval.json").exists() else {}
if E:
    stats = {l: json.load(open(cfg.data / "counts" / f"{l}.json"))["stats"] for l in cfg.langs}
    w = np.array([stats[l]["chars"] for l in cfg.langs], float); w /= w.sum()
    comp = lambda e: float(np.dot(w, [e["per_lang"][l]["T"] for l in cfg.langs]))
    held = {}
    for sw, bkey, ykey in [("parity", "tau", "T_max"), ("morph", "eps", "spurious_mean")]:
        R = sorted([r for r in recs if r["sweep"] == sw], key=lambda r: -r[bkey])
        R = [r for r in R if f"toast_{r['tag']}" in E]
        if not R:
            continue
        xs = [comp(E[f"toast_{r['tag']}"]) for r in R]; ys = [E[f"toast_{r['tag']}"].get(ykey, np.nan) for r in R]
        held[sw] = {"tags": [r["tag"] for r in R], "T_weighted": xs, ykey: ys,
                    "T_weighted_monotone": bool(np.all(np.diff(xs) >= 0)), f"{ykey}_monotone": bool(np.all(np.diff(ys) <= 0))}
        for k in ["T_weighted_monotone", f"{ykey}_monotone"]:
            if not held[sw][k]:
                flags.append(f"{sw}: held-out {k} is False: T={np.round(xs, 2)}, {ykey}={np.round(ys, 3)}")
    out["monotone_heldout"] = held

# ---- 3. round trip on all held-out lines + unseen characters
probe = ["Hello 🌍 wörld — ¿qué? 漢字テスト ✓", "\t  tabs\n\nnewlines  ", "Ελληνικά ქართული ꯃꯩꯇꯩ", "‍zwj nbsp"]
eval_texts = {l: read_lines(cfg.data / "eval" / f"{l}.txt", a.max_lines) for l in cfg.langs}
rt = {}
for p in sorted((cfg.out / "tokenizers").iterdir()):
    tok = load_any(p)
    if tok is None:
        continue
    bad = [(l, t) for l, ts in eval_texts.items() for t in ts if tok.decode_tokens(t) != t]
    bad += [("probe", t) for t in probe if tok.decode_tokens(t) != t]
    n = sum(map(len, eval_texts.values())) + len(probe)
    rt[p.stem] = {"checked": n, "failures": len(bad), "examples": [f"{l}: {t[:60]!r}" for l, t in bad[:3]]}
    if bad:
        hard_fail.append(f"{p.stem}: {len(bad)}/{n} round-trip failures, e.g. {rt[p.stem]['examples'][0]}")
out["roundtrip"] = rt

out["hard_failures"], out["flags"] = hard_fail, flags
json.dump(out, open(cfg.out / "checks.json", "w"), indent=1, default=float)
print(f"LP <= integer: {sum(r['lp_le_int'] for r in rows)}/{len(rows)} points; "
      f"integer bound violations at {sum(bool(r['int_bound_viol']) for r in rows)} points")
for sw, d in mono.items():
    print(f"train-side {sw}: LP monotone={d['lp_monotone']}, " + ", ".join(f"{k}={v}" for k, v in d.items() if k.startswith("int_") and k.endswith("monotone")))
for sw, d in out.get("monotone_heldout", {}).items():
    print(f"held-out  {sw}: " + ", ".join(f"{k}={v}" for k, v in d.items() if k.endswith("monotone")))
print(f"round trip: {sum(v['failures'] for v in rt.values())} failures over {len(rt)} tokenizers "
      f"x {next(iter(rt.values()))['checked'] if rt else 0} strings")
for f in flags:
    print("FLAG:", f)
for f in hard_fail:
    print("FAIL:", f)
sys.exit(1 if hard_fail else 0)
