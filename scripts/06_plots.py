#!/usr/bin/env python3
"""Figures for slides / report.

  python scripts/06_plots.py --config configs/smoke.yaml

  fig_parity_front.png   held-out: compression (data-weighted T) vs worst-language T, LP sweep vs BPE/Unigram
  fig_morph_front.png    held-out: compression vs spurious boundaries/word on gold morphology
  fig_per_lang.png       per-language tokens/1000 chars for selected tokenizers (the motivation figure)
  fig_shadow_prices.png  LP: d(Ttot)/d(bound) along each sweep (exchange rate between objectives)
  summary.md             table of all tokenizers + hypervolumes
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from motok.config import load_cfg
from motok.eval.metrics import hypervolume, nondominated

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
cfg = load_cfg(a.config)
E = json.load(open(cfg.out / "eval.json"))
S = json.load(open(cfg.out / "sweep.json")) if (cfg.out / "sweep.json").exists() else {"records": []}
langs = cfg.langs
stats = {l: json.load(open(cfg.data / "counts" / f"{l}.json"))["stats"] for l in langs}
w = np.array([stats[l]["chars"] for l in langs], float); w /= w.sum()


def comp(r):   # data-weighted tokens / 1000 chars on held-out text (what plain compression optimises)
    return float(np.dot(w, [r["per_lang"][l]["T"] for l in langs]))


def group(name):
    if name.startswith("bpe"): return "BPE (temp. sampling)"
    if name.startswith("unigram"): return "Unigram (temp. sampling)"
    if "parity" in name or name.endswith("minmax"): return "LP: compression vs parity"
    if "morph" in name: return "LP: compression vs morphology"
    if "grid" in name: return "LP: 3-objective grid"
    return "LP anchors"


style = {"BPE (temp. sampling)": ("s", "tab:red"), "Unigram (temp. sampling)": ("^", "tab:orange"),
         "LP: compression vs parity": ("o", "tab:blue"), "LP: compression vs morphology": ("D", "tab:green"),
         "LP: 3-objective grid": (".", "tab:gray"), "LP anchors": ("*", "black")}


bound = {f"toast_{r['tag']}": r.get("tau", r.get("eps")) for r in S["records"]
         if r["sweep"] in ("parity", "morph") and "lp" in r}


def front_plot(ykey, ylabel, fname, groups_on):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    pts = {}
    for n, r in E.items():
        g = group(n)
        if g not in groups_on or (ykey == "spurious_mean" and "spurious_mean" not in r):
            continue
        y = r["T_max"] if ykey == "T_max" else r[ykey]
        pts.setdefault(g, []).append((comp(r), y, n))
    for g, P in pts.items():
        m, c = style[g]
        if g.startswith("LP:"):
            # connect in sweep order (compression anchor -> loose -> tight -> min-max / min-morph anchor),
            # not by x: held-out values need not be monotone along the sweep
            P = sorted(P, key=lambda p: (0, -bound[p[2]]) if p[2] in bound else (1, 0))
            if "toast_compression" in E:
                anc = E["toast_compression"]
                xa, ya = comp(anc), anc["T_max"] if ykey == "T_max" else anc.get(ykey)
                ax.plot([xa] + [p[0] for p in P], [ya] + [p[1] for p in P], "-", color=c, alpha=.5)
            ax.plot([p[0] for p in P], [p[1] for p in P], m, color=c, label=g + " (line = sweep order)", ms=6, alpha=.9)
        else:
            ax.plot([p[0] for p in P], [p[1] for p in P], m, color=c, label=g, ms=9 if m == "*" else 6, alpha=.9)
        for x, y, n in P:
            if g in ("LP anchors",) or not g.startswith("LP"):
                ax.annotate(n.replace("toast_", ""), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("compression: data-weighted tokens / 1000 chars (held-out, lower = better)")
    ax.set_ylabel(ylabel); ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(cfg.out / fname, dpi=160); plt.close(fig)
    return pts


p1 = front_plot("T_max", "worst-language tokens / 1000 chars (lower = fairer)", "fig_parity_front.png",
                ["BPE (temp. sampling)", "Unigram (temp. sampling)", "LP: compression vs parity", "LP anchors"])
p2 = front_plot("spurious_mean", "spurious boundaries / word, gold (lower = better)",
                "fig_morph_front.png",
                ["BPE (temp. sampling)", "Unigram (temp. sampling)", "LP: compression vs morphology", "LP anchors"])

# per-language bars
pick = [n for n in ["bpe_a1.0", "bpe_a0.0", "toast_compression", "toast_minmax"] if n in E]
fig, ax = plt.subplots(figsize=(7, 3.8))
bw = 0.8 / max(len(pick), 1)
for i, n in enumerate(pick):
    ax.bar(np.arange(len(langs)) + i * bw, [E[n]["per_lang"][l]["T"] for l in langs], bw, label=n)
ax.set_xticks(np.arange(len(langs)) + bw * (len(pick) - 1) / 2); ax.set_xticklabels(langs)
ax.set_ylabel("tokens / 1000 chars (held-out)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(cfg.out / "fig_per_lang.png", dpi=160); plt.close(fig)

# shadow prices along sweeps
recs = S["records"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, sw, key, bkey, lab in [(axs[0], "parity", "t", "tau", "worst-language bound tau"),
                               (axs[1], "morph", "Mtot", "eps", "morph bound eps (spurious / 1000 chars)")]:
    R = sorted([r for r in recs if r["sweep"] == sw], key=lambda r: r[bkey])
    if R:
        ax.plot([r[bkey] for r in R], [-r["rc"][key] for r in R], "o-")
        ax.set_xlabel(lab); ax.set_ylabel("-d(Ttot)/d(bound)  [shadow price]"); ax.grid(alpha=.3)
        ax.set_title(f"{sw}: tokens saved per unit of relaxation", fontsize=9)
fig.tight_layout(); fig.savefig(cfg.out / "fig_shadow_prices.png", dpi=160); plt.close(fig)

# summary table + hypervolumes (2-D, held-out)
lines = ["| tokenizer | T weighted | T max | Gini(T) | Rényi eff | morph F1 | spurious/word | roundtrip errs |",
         "|---|---|---|---|---|---|---|---|"]
for n, r in sorted(E.items()):
    lines.append(f"| {n} | {comp(r):.1f} | {r['T_max']:.1f} | {r['T_gini']:.3f} | {r['renyi_eff']:.3f} | "
                 f"{r.get('morph_f1_mean', float('nan')):.3f} | {r.get('spurious_mean', float('nan')):.3f} | "
                 f"{r.get('roundtrip_failures', 0)} |")
allpts = [(comp(r), r["T_max"]) for r in E.values()]
ref = np.max(allpts, 0) * 1.05
lines.append("\n**Hypervolume (compression × worst-language, held-out, ref = 1.05 × max):**\n")
for g, P in p1.items():
    A = np.array([(x, y) for x, y, _ in P])
    lines.append(f"- {g}: {hypervolume(A[nondominated(A)], ref):.1f}")
(cfg.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"figures -> {cfg.out}")
