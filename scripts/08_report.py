#!/usr/bin/env python3
"""Final figures + tables for the report.

  python scripts/08_report.py --config configs/full.yaml --lp full_20k [--lp2 full_10k] [--scaling full full_10k full_20k full_40k]

Reads  results/full/eval.json      held-out metrics of every tokenizer (baselines + LP points linked as lp20k_*, lp10k_*)
       results/<lp>/sweep.json     training-side LP fronts, gaps, shadow prices, timings
       results/full/lm/*.json      small-LM bits per byte
Writes results/final/{fig_*.png, tables.md, numbers.json}
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from motok.config import ROOT, load_cfg, read_lines
from motok.eval.metrics import hypervolume, nondominated

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--lp", default="full_10k_t10k:lp10t", help="main LP run as <results dir>:<prefix in eval.json>")
ap.add_argument("--others", nargs="*", default=["full_10k:lp10k", "full_5k_t5k:lp5t"],
                help="further LP runs drawn lightly in the held-out fronts, <dir>:<prefix>")
ap.add_argument("--scaling", nargs="+", default=["full", "full_10k", "full_20k", "full_40k"])
a = ap.parse_args()
cfg = load_cfg(a.config)
OUT = ROOT / "results" / "final"
OUT.mkdir(parents=True, exist_ok=True)
langs = cfg.langs
E = json.load(open(cfg.out / "eval.json"))
LP_DIR, P1 = a.lp.split(":"); P1 += "_"
OTHERS = [(d, p + "_") for d, p in (x.split(":") for x in a.others)]
NAMES = {"lp10t": "LP SAA 10k+10k", "lp5t": "LP SAA 5k+5k", "lp10k": "LP head-only 10k", "lp10km": "LP morph-first 10k",
         "lp20t": "LP SAA 20k+20k", "lp20k": "LP head-only 20k"}
nm = lambda pf: NAMES.get(pf.rstrip("_"), pf.rstrip("_"))
pref = {LP_DIR: P1, **{d: p for d, p in OTHERS}}
stats = {l: json.load(open(cfg.data / "counts" / f"{l}.json"))["stats"] for l in langs}
w = np.array([stats[l]["chars"] for l in langs], float); w /= w.sum()
n_flores = len(read_lines(cfg.data / "flores" / "devtest" / "en.txt"))

# ---- palette (reference instance, light mode) + marker shapes as secondary encoding
C = {"lp": "#2a78d6", "bpe": "#eb6834", "uni": "#1baf7a", "par": "#6250d6", "mor": "#e87ba4",
     "lpw": "#2a78d6", "anchor": "#0b0b0b", "lp2": "#8fb4e6", "grid": "#9b9a96"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 180, "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def T_w(r):      # data-weighted tokens / 1000 English-equivalent chars on held-out text (= the LP's Ttot)
    return float(np.dot(w, [r["per_lang"][l]["T"] for l in langs]))


def T_mean(r):
    return float(np.mean([r["per_lang"][l]["T"] for l in langs]))


def flores_tok(r):  # tokens per FLORES sentence, per language
    # premium = tokens_l / tokens_en; recover absolute counts via the English T is not possible, so eval stores premium
    return r.get("flores_tokens")


def family(n):
    if n.startswith("bpe_"): return "bpe"
    if n.startswith("unigram_"): return "uni"
    if n.startswith("paritybpe"): return "par"
    if n.startswith("morphbpe"): return "mor"
    if n.startswith(P1): return "lp"
    if any(n.startswith(p) for p in pref.values()): return "lp2"
    if n.startswith("lp"): return "lpx"
    return "other"


def lp_tag(n):
    for p in pref.values():
        if n.startswith(p):
            return n[len(p):]
    return n


S = json.load(open(ROOT / "results" / LP_DIR / "sweep.json"))
recs = {r["tag"]: r for r in S["records"] if "lp" in r}
numbers = {"lp": LP_DIR, "lp_size": {k: S[k] for k in ["n_tokens", "n_nodes", "n_rows", "n_trees", "build_seconds"]},
           "coverage": S["coverage"]}

# =========================================================== 1. held-out fronts (FineWeb eval + FLORES)
def series(pred, key):
    return sorted([(n, r) for n, r in E.items() if pred(n)], key=key)


def front_axes(ax, xf, yf, xlabel, ylabel, sweep="parity"):
    lp_names = [n for n in E if family(n) == "lp"]
    sw = sorted([n for n in lp_names if lp_tag(n).startswith(sweep + "_")],
                key=lambda n: -recs.get(lp_tag(n), {}).get("tau" if sweep == "parity" else "eps", 0))
    anchor = P1 + "compression"
    end = P1 + ("minmax" if sweep == "parity" else "min_morph")
    chain = [n for n in [anchor] + sw + [end] if n in E]
    ax.plot([xf(E[n]) for n in chain], [yf(E[n]) for n in chain], "-o", color=C["lp"], lw=2, ms=5,
            label=f"{nm(P1)}: ε-constraint ({sweep})", zorder=3)
    wn = sorted([n for n in lp_names if lp_tag(n).startswith("wpar_" if sweep == "parity" else "wmor_")])
    if wn:
        ax.plot([xf(E[n]) for n in wn], [yf(E[n]) for n in wn], "s", mfc="none", mec=C["lpw"], ms=6, mew=1.2,
                label=f"{nm(P1)}: weighted sum", zorder=3)
    for (d, pf), col in zip([o for o in OTHERS if not o[1].startswith("lp10km")][:2], ["#8fb4e6", "#b9b8b2"]):
        R2 = json.load(open(ROOT / "results" / d / "sweep.json"))["records"] if (ROOT / "results" / d / "sweep.json").exists() else []
        b2 = {r["tag"]: r.get("tau" if sweep == "parity" else "eps", 0) for r in R2}
        ch = [pf + "compression"] + sorted([n[len(pf):] and n for n in E if n.startswith(pf + sweep + "_")],
                                            key=lambda n: -b2.get(n[len(pf):], 0)) + [pf + ("minmax" if sweep == "parity" else "min_morph")]
        ch = [n for n in ch if n in E]
        if ch:
            ax.plot([xf(E[n]) for n in ch], [yf(E[n]) for n in ch], "-", color=col, lw=1.5, marker=".", ms=4,
                    label=f"{nm(pf)} ({sweep} sweep)", zorder=1)
    for n, lab in [(anchor, "LP compression"), (end, "LP " + ("min-max" if sweep == "parity" else "min-morph"))]:
        if n in E:
            ax.plot(xf(E[n]), yf(E[n]), "*", color=C["anchor"], ms=11, zorder=4)
            ax.annotate(lab, (xf(E[n]), yf(E[n])), xytext=(6, 5) if "compression" in lab else (6, -11),
                        textcoords="offset points", fontsize=7, color=INK2)
    for fam, lab, mk, pat in [("bpe", "BPE (temperature α)", "D", "bpe_a"), ("uni", "Unigram (temperature α)", "^", "unigram_a")]:
        pts = series(lambda n: n.startswith(pat), key=lambda t: float(t[0].split("_a")[1]))
        if pts:
            ax.plot([xf(r) for _, r in pts], [yf(r) for _, r in pts], "-" + mk, color=C[fam], lw=1.5, ms=6, label=lab, zorder=2)
            for n, r in ([pts[0], pts[-1]] if fam == "bpe" else [pts[0]]):
                ax.annotate("α=" + n.split("_a")[1], (xf(r), yf(r)), xytext=(4, -10), textcoords="offset points",
                            fontsize=7, color=INK2)
    for n, lab, fam, mk in [("paritybpe_base", "Parity-aware BPE", "par", "P"), ("morphbpe", "MorphBPE", "mor", "X")]:
        if n in E:
            ax.plot(xf(E[n]), yf(E[n]), mk, color=C[fam], ms=9, label=lab, zorder=4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def flores_abs(r):
    return r.get("flores_abs")   # tokens per sentence, per language (added by 05_evaluate --flores-abs)


fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
front_axes(axs[0], T_w, lambda r: r["T_max"],
           "compression: data-weighted tokens / 1000 chars (held-out FineWeb)",
           "worst-language tokens / 1000 chars")
axs[0].set_title("(a) held-out FineWeb text", fontsize=9, loc="left")
have_fl = all("flores_abs" in r for r in E.values())
if have_fl:
    front_axes(axs[1], lambda r: np.mean(list(r["flores_abs"].values())), lambda r: max(r["flores_abs"].values()),
               "mean tokens / FLORES sentence (same content in all languages)", "worst-language tokens / sentence")
    axs[1].set_title("(b) FLORES+ devtest (parallel)", fontsize=9, loc="left")
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
fig.savefig(OUT / "fig_parity_front.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
front_axes(axs[0], T_w, lambda r: r["ms_spurious_mean"], "compression: data-weighted tokens / 1000 chars (held-out)",
           "spurious boundaries / word (MorphScore gold, 6 langs)", sweep="morph")
axs[0].set_title("(a) spurious morpheme boundaries", fontsize=9, loc="left")
front_axes(axs[1], T_w, lambda r: r["ms_f1_mean"], "compression: data-weighted tokens / 1000 chars (held-out)",
           "boundary F1 (MorphScore gold, higher = better)", sweep="morph")
axs[1].set_title("(b) morpheme-boundary F1", fontsize=9, loc="left")
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
fig.savefig(OUT / "fig_morph_front.png"); plt.close(fig)

# =========================================================== 2. training-side fronts with certificates
def train_fronts(R_all, fname, title):
    rc_ = {r["tag"]: r for r in R_all if "lp" in r}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, sw, wsw, yk, ylab in [(axs[0], "parity", "wpar", "Tmax", "worst-language T (training side)"),
                                  (axs[1], "morph", "wmor", "Mtot", "spurious boundaries / 1000 chars (training side)")]:
        end = "minmax" if sw == "parity" else "min_morph"
        R = [rc_[t] for t in ["compression"] + sorted([t for t in rc_ if t.startswith(sw + "_")], key=lambda t: int(t.split("_")[1])) + [end] if t in rc_]
        ax.plot([r["lp"]["Ttot"] for r in R], [r["lp"][yk] for r in R], "-", color=C["lp"], lw=2, label="LP relaxation (lower bound)")
        ax.plot([r["int"]["Ttot"] for r in R], [r["int"][yk] for r in R], "o", color=C["lp"], ms=5, mfc="#fcfcfb", mew=1.5,
                label="rounded integer vocabulary")
        Wr = [rc_[t] for t in sorted([t for t in rc_ if t.startswith(wsw + "_")], key=lambda t: int(t.split("_")[1]))]
        if Wr:
            ax.plot([r["int"]["Ttot"] for r in Wr], [r["int"][yk] for r in Wr], "s", mfc="none", mec=C["bpe"], ms=6,
                    label="weighted-sum points (integer)")
        G = [r for t, r in rc_.items() if t.startswith("grid_")]
        if G and sw == "parity":
            ax.plot([r["int"]["Ttot"] for r in G], [r["int"][yk] for r in G], ".", color=C["grid"], ms=6, label="3-objective grid")
        ax.set_xlabel("Ttot: data-weighted tokens / 1000 chars (training side)"); ax.set_ylabel(ylab)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.legend(fontsize=7)
    axs[0].set_title(f"(a) {title}: compression vs worst language", fontsize=9, loc="left")
    axs[1].set_title(f"(b) {title}: compression vs morphology", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


train_fronts(S["records"], "fig_train_fronts.png", nm(P1))
for d, pf in OTHERS[:1]:
    if (ROOT / "results" / d / "sweep.json").exists():
        train_fronts(json.load(open(ROOT / "results" / d / "sweep.json"))["records"], f"fig_train_fronts_{pf.rstrip('_')}.png",
                     nm(pf))

# shadow prices
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
for ax, sw, key, bkey, lab in [(axs[0], "parity", "t", "tau", "worst-language bound τ"),
                               (axs[1], "morph", "Mtot", "eps", "morph bound ε (spurious / 1000 chars)")]:
    R = sorted([r for t, r in recs.items() if t.startswith(sw + "_")], key=lambda r: r[bkey])
    if R:
        ax.plot([r[bkey] for r in R], [-r["rc"][key] for r in R], "-o", color=C["lp"], lw=2, ms=5)
        ax.set_xlabel(lab); ax.set_ylabel("−dTtot / d(bound)")
        ax.set_title(f"{sw}: total tokens saved per unit of relaxation", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_shadow_prices.png"); plt.close(fig)

# =========================================================== 3. per-language bars (FLORES tokens / sentence)
pick = [n for n in ["bpe_a1.0", "bpe_a0.0", "paritybpe_base", P1 + "compression", P1 + "minmax"] if n in E]
if have_fl and pick:
    cols = [C["bpe"], "#c94f1f", C["par"], C["lp"], "#173f73"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bw = 0.8 / len(pick)
    for i, n in enumerate(pick):
        ax.bar(np.arange(len(langs)) + i * bw - 0.4 + bw / 2, [E[n]["flores_abs"][l] for l in langs], bw * 0.92,
               color=cols[i % len(cols)], label=n.replace(P1, nm(P1) + " "))
    ax.set_xticks(np.arange(len(langs))); ax.set_xticklabels(langs)
    ax.set_ylabel("tokens / FLORES sentence"); ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_per_lang.png"); plt.close(fig)

# =========================================================== 4. scaling
sc = []
for name in a.scaling:
    d = ROOT / "results" / name
    ps = d / "problem_stats.json"
    if not ps.exists():
        continue
    st = json.load(open(ps))
    row = {"name": name, "trees_per_lang": st.get("trees_per_lang", 5000), "n_trees": st["n_trees"], "n_nodes": st["n_nodes"],
           "coverage_min": min(st["coverage"].values()), "coverage_mean": float(np.mean(list(st["coverage"].values()))),
           "build_seconds": st.get("build_seconds")}
    for an in ["compression", "minmax", "min_morph"]:
        f = d / "sweep_parts" / f"anchor_{an}.json"
        if f.exists():
            rr = json.load(open(f))["records"][0]
            row[f"t_{an}"] = rr["seconds"]; row[f"gap_{an}"] = rr.get("gap_Ttot")
    sj = d / "sweep.json"
    if sj.exists():
        R = [r for r in json.load(open(sj))["records"] if r["sweep"] not in ("anchor",) and "lp" in r]
        if R:
            row["warm_median_s"] = float(np.median([r["seconds"] for r in R]))
            row["n_points"] = len(R)
            row["max_abs_gap"] = float(max(abs(r["gap_Ttot"]) for r in R if r["sweep"] not in ("wpar", "wmor")) if R else 0)
    sc.append(row)
numbers["scaling"] = sc
if sc:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    x = [r["n_nodes"] / 1e6 for r in sc]
    for key, lab, mk in [("t_compression", "compression anchor (cold)", "o"), ("t_minmax", "min-max anchor (cold)", "s"),
                         ("warm_median_s", "sweep point (warm, median)", "^")]:
        pts = [(xx, r[key]) for xx, r in zip(x, sc) if r.get(key)]
        if pts:
            axs[0].plot(*zip(*pts), "-" + mk, lw=2, ms=6, label=lab,
                        color={"t_compression": C["lp"], "t_minmax": C["bpe"], "warm_median_s": C["uni"]}[key])
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    axs[0].xaxis.set_major_locator(FixedLocator(x)); axs[0].xaxis.set_minor_locator(NullLocator())
    axs[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axs[0].set_xlabel("LP node variables (millions)"); axs[0].set_ylabel("solve time (s)"); axs[0].legend(fontsize=7, loc="lower right")
    axs[1].plot([r["trees_per_lang"] / 1e3 for r in sc], [100 * r["coverage_mean"] for r in sc], "-o", color=C["lp"], lw=2, label="mean over languages")
    axs[1].plot([r["trees_per_lang"] / 1e3 for r in sc], [100 * r["coverage_min"] for r in sc], "-s", color=C["bpe"], lw=2, label="worst language")
    axs[1].set_xlabel("trees per language (thousands)"); axs[1].set_ylabel("pretoken mass covered by trees (%)")
    axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_scaling.png"); plt.close(fig)

# =========================================================== 5. small-LM bits per byte
LM = {}
for f in sorted((cfg.out / "lm").glob("*_s*.json")):
    d = json.load(open(f))
    if d["args"].get("epochs", 1) < 4:
        continue
    LM.setdefault(d["tokenizer"], []).append(d)
lm_rows = []
for tok, runs in LM.items():
    fl_bits = {l: [r["bpb"][f"flores_{l}"] for r in runs] for l in langs}
    row = {"tokenizer": tok, "seeds": len(runs),
           "fw_bpb_mean": float(np.mean([r["fw_bpb_mean"] for r in runs])),
           "fw_bpb_sd": float(np.std([r["fw_bpb_mean"] for r in runs])),
           "flores_bpb_mean": float(np.mean([r["flores_bpb_mean"] for r in runs])),
           "flores_bpb_sd": float(np.std([r["flores_bpb_mean"] for r in runs])),
           "per_lang_fw": {l: float(np.mean([r["bpb"][f"fw_{l}"] for r in runs])) for l in langs},
           "per_lang_flores": {l: float(np.mean(v)) for l, v in fl_bits.items()},
           "train_tokens": runs[0]["train_tokens"], "steps": runs[0]["steps"]}
    # bits per FLORES sentence (content-matched): bpb * bytes / n_sentences
    fb = {l: json.load(open(cfg.out / "lm" / "data" / tok / "meta.json"))["eval"][f"flores_{l}"]["bytes"] for l in langs}
    row["bits_per_sent"] = {l: row["per_lang_flores"][l] * fb[l] / n_flores for l in langs}
    row["bits_per_sent_max"] = max(row["bits_per_sent"].values())
    row["bits_per_sent_mean"] = float(np.mean(list(row["bits_per_sent"].values())))
    lm_rows.append(row)
lm_rows.sort(key=lambda r: r["fw_bpb_mean"])
numbers["lm"] = lm_rows
if lm_rows:
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    names = [r["tokenizer"] for r in lm_rows]
    fam_col = lambda n: C.get(family(n), C["lp"]) if family(n) != "lp2" else C["lp2"]
    bw = 0.8 / len(langs)
    for i, l in enumerate(langs):
        pass
    y = np.arange(len(names))
    axs[0].barh(y, [r["fw_bpb_mean"] for r in lm_rows], xerr=[r["fw_bpb_sd"] for r in lm_rows], color=[fam_col(n) for n in names],
                height=0.7, error_kw={"ecolor": INK2, "lw": 1})
    axs[0].set_yticks(y); axs[0].set_yticklabels([n.replace(P1, "LP ") for n in names], fontsize=7)
    axs[0].invert_yaxis(); axs[0].set_xlabel("bits per byte, held-out FineWeb (mean of 6 langs, ±sd over seeds)")
    lo = min(r["fw_bpb_mean"] for r in lm_rows); hi = max(r["fw_bpb_mean"] for r in lm_rows)
    axs[0].set_xlim(lo - 0.6 * (hi - lo) - 0.01, hi + 0.3 * (hi - lo) + 0.01)
    axs[0].grid(axis="y", visible=False)
    for r in lm_rows:
        tk = E.get(r["tokenizer"])
        if tk and have_fl:
            axs[1].plot(np.mean(list(tk["flores_abs"].values())), r["bits_per_sent_max"], "o", color=fam_col(r["tokenizer"]), ms=7)
            axs[1].annotate(r["tokenizer"].replace(P1, "LP "), (np.mean(list(tk["flores_abs"].values())), r["bits_per_sent_max"]),
                            xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    axs[1].set_xlabel("mean tokens / FLORES sentence (tokenizer)")
    axs[1].set_ylabel("worst-language bits / FLORES sentence (LM)")
    fig.tight_layout(); fig.savefig(OUT / "fig_lm.png"); plt.close(fig)

# =========================================================== tables
L = []
L.append("## Held-out intrinsic metrics (FineWeb eval split; FLORES+ devtest; MorphScore / SIGMORPHON gold)\n")
L.append("| tokenizer | T weighted | T mean | T max | Gini(T) | FLORES tok/sent mean | FLORES max | max premium | Rényi | MorphScore F1 | spur/word (MS) | SIGM. F1 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
order = lambda n: ({"bpe": 0, "uni": 1, "par": 2, "mor": 3, "lp": 4, "lp2": 5}.get(family(n), 9), n)
for n in sorted(E, key=order):
    r = E[n]
    fa = r.get("flores_abs", {})
    L.append(f"| {n} | {T_w(r):.1f} | {T_mean(r):.1f} | {r['T_max']:.1f} | {r['T_gini']:.3f} | "
             f"{np.mean(list(fa.values())) if fa else float('nan'):.1f} | {max(fa.values()) if fa else float('nan'):.1f} | "
             f"{r.get('premium_max', float('nan')):.2f} | {r['renyi_eff']:.3f} | {r.get('ms_f1_mean', float('nan')):.3f} | "
             f"{r.get('ms_spurious_mean', float('nan')):.3f} | {r.get('morph_f1_mean', float('nan')):.3f} |")

# hypervolumes on (T weighted, T max) and (FLORES mean, FLORES max), per family (union of points per family)
def hv_table(xf, yf, title):
    fams = {f"{nm(P1)} (main run, all sweep points)": [n for n in E if family(n) == "lp"], "BPE (α sweep)": [n for n in E if family(n) == "bpe"],
            "Unigram (α sweep)": [n for n in E if family(n) == "uni"],
            "all baselines": [n for n in E if family(n) in ("bpe", "uni", "par", "mor")],
            **{f"{nm(pf)} (comparison)": [n for n in E if n.startswith(pf)] for _, pf in OTHERS}}
    allp = np.array([(xf(E[n]), yf(E[n])) for n in E if family(n) in ("lp", "bpe", "uni", "par", "mor")])
    ideal, ref = allp.min(0), allp.max(0) * 1.02
    out = [f"\n**Hypervolume — {title}** (normalised to the box [ideal, 1.02×nadir]; higher = better)\n"]
    for k, ns in fams.items():
        if ns:
            A = np.array([(xf(E[n]), yf(E[n])) for n in ns])
            hv = hypervolume(A[nondominated(A)], ref) / np.prod(ref - ideal)
            out.append(f"- {k}: {hv:.3f}  ({len(ns)} tokenizers)")
            numbers.setdefault("hv", {}).setdefault(title, {})[k] = hv
    return out


L += hv_table(T_w, lambda r: r["T_max"], "held-out compression × worst language")
if have_fl:
    L += hv_table(lambda r: np.mean(list(r["flores_abs"].values())), lambda r: max(r["flores_abs"].values()),
                  "FLORES mean × worst tokens/sentence")
L += hv_table(T_w, lambda r: r["ms_spurious_mean"], "held-out compression × spurious boundaries (MorphScore)")

L.append(f"\n## LP sweep ({LP_DIR}, training side; vocab {S['vocab_size']}, {S['n_trees']} trees, {S['n_nodes']} nodes, {S['n_rows']} rows)\n")
L.append("| point | bound | LP Ttot | int Ttot | int Tmax | int Mtot | gap(Ttot) | int feasible | shade | frac x/z | −rc | time (s) | iters |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in S["records"]:
    if "lp" not in r:
        L.append(f"| {r['tag']} | {r.get('tau', r.get('eps', ''))} | {r['status']} | | | | | | | | | {r['seconds']:.0f} | {r['iters']} |")
        continue
    b = r.get("tau", r.get("eps", r.get("lam", r.get("mu", ""))))
    rc = -r["rc"]["t"] if r["sweep"] == "parity" else (-r["rc"]["Mtot"] if r["sweep"] == "morph" else float("nan"))
    L.append(f"| {r['tag']} | {b if b == '' else f'{b:.3g}'} | {r['lp']['Ttot']:.2f} | {r['int']['Ttot']:.2f} | {r['int']['Tmax']:.2f} | "
             f"{r['int']['Mtot']:.2f} | {r['gap_Ttot']:.1e} | {r.get('int_feasible', '')} | {r.get('shade_steps', 0)} | "
             f"{r['n_frac_x']}/{r['n_frac_z']} | {rc:.3g} | {r['seconds']:.0f} | {r['iters']} |")

if sc:
    L.append("\n## Scaling (trees per language)\n")
    L.append("| run | trees/lang | trees | nodes | coverage mean / min | build (s) | compression (s) | min-max (s) | min-morph (s) | warm point median (s) | max |gap| |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sc:
        f = lambda k: f"{r[k]:.0f}" if r.get(k) else "–"
        L.append(f"| {r['name']} | {r['trees_per_lang']} | {r['n_trees']} | {r['n_nodes']} | {100*r['coverage_mean']:.1f}% / {100*r['coverage_min']:.1f}% | "
                 f"{f('build_seconds')} | {f('t_compression')} | {f('t_minmax')} | {f('t_min_morph')} | {f('warm_median_s')} | "
                 f"{r.get('max_abs_gap', float('nan')):.1e} |")

if lm_rows:
    L.append("\n## Small LM (35.5M params, 18.9M non-embedding; equal bytes per language; 4 epochs)\n")
    L.append("| tokenizer | seeds | train tokens | steps | FineWeb bpb (mean ± sd) | FLORES bpb | bits/FLORES sent. mean | worst | " + " | ".join(f"bpb {l}" for l in langs) + " |")
    L.append("|---|---|---|---|---|---|---|---|" + "---|" * len(langs))
    for r in lm_rows:
        L.append(f"| {r['tokenizer']} | {r['seeds']} | {r['train_tokens']/1e6:.1f}M | {r['steps']} | {r['fw_bpb_mean']:.4f} ± {r['fw_bpb_sd']:.4f} | "
                 f"{r['flores_bpb_mean']:.4f} | {r['bits_per_sent_mean']:.1f} | {r['bits_per_sent_max']:.1f} | "
                 + " | ".join(f"{r['per_lang_fw'][l]:.3f}" for l in langs) + " |")

(OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
json.dump(numbers, open(OUT / "numbers.json", "w"), indent=1, default=float)
print("\n".join(L))
print(f"-> {OUT}")
