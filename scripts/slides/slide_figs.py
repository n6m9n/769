"""Slide-sized figures (true font sizes at their slot size) from the copied result JSONs."""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(D, "results")
OUT = "/Users/naman/Desktop/769/motok/report/slide_figures/"
os.makedirs(OUT, exist_ok=True)
LANGS = ["en", "hi", "mr", "ta", "ru", "hu"]

E = json.load(open(f"{R}/full/eval.json"))
S10t = json.load(open(f"{R}/full_10k_t10k/sweep.json"))
S10k = json.load(open(f"{R}/full_10k/sweep.json"))
chars = {}
for line in open(f"{D}/counts_stats.txt"):
    l, js = line.split(" ", 1)
    chars[l.replace(".json", "")] = json.loads(js[: js.rindex("}") + 1])["chars"]
w = np.array([chars[l] for l in LANGS], float); w /= w.sum()

INK, INK2, GRID, BG = "#1E1E24", "#55575f", "#e6e7eb", "#ffffff"
LP, LPL, BPE, UNI, PAR, MOR, GREY = "#2a78d6", "#9cc0ee", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#b4b6bd"
plt.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "cm", "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.06, "figure.facecolor": BG, "axes.facecolor": BG,
    "lines.linewidth": 2.4,
})

Tw = lambda r: float(np.dot(w, [r["per_lang"][l]["T"] for l in LANGS]))
flm = lambda r: float(np.mean([r["flores_abs"][l] for l in LANGS]))
flx = lambda r: float(max(r["flores_abs"].values()))
NICE = {"bpe_a1.0": "BPE (natural mix)", "bpe_a0.0": "BPE (uniform mix)", "paritybpe_base": "Parity-aware BPE",
        "morphbpe": "MorphBPE", "unigram_a1.0": "Unigram (natural mix)", "lp10t_compression": "LP: compression anchor",
        "lp10t_minmax": "LP: min-max anchor", "lp10t_parity_5": "LP: parity point"}


def save(fig, name):
    fig.savefig(OUT + name); plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- 1. per-language FLORES bars
pick = ["bpe_a1.0", "bpe_a0.0", "paritybpe_base", "lp10t_compression", "lp10t_minmax"]
cols = [BPE, "#b8481c", PAR, LPL, LP]
fig, ax = plt.subplots(figsize=(7.7, 4.15))
bw = 0.16
for i, n in enumerate(pick):
    ax.bar(np.arange(6) + (i - 2) * bw, [E[n]["flores_abs"][l] for l in LANGS], bw * 0.92, color=cols[i], label=NICE[n])
ax.set_xticks(range(6)); ax.set_xticklabels(["English", "Hindi", "Marathi", "Tamil", "Russian", "Hungarian"])
ax.set_ylabel("tokens per FLORES+ sentence"); ax.set_ylim(0, 52); ax.grid(axis="x", visible=False)
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=11.5, columnspacing=1.0, handlelength=1.2)
save(fig, "s_perlang.png")


# ---------------------------------------------------------------- 2/5. training-side fronts
def train_front(S, name, size=(12.0, 4.0)):
    rc = {r["tag"]: r for r in S["records"] if "lp" in r}
    fig, axs = plt.subplots(1, 2, figsize=size)
    for ax, sw, wsw, yk, ylab in [(axs[0], "parity", "wpar", "Tmax", r"worst-language cost $t$"),
                                  (axs[1], "morph", "wmor", "Mtot", r"spurious boundaries $M_{\mathrm{tot}}$")]:
        end = "minmax" if sw == "parity" else "min_morph"
        R_ = [rc[t] for t in ["compression"] + sorted([t for t in rc if t.startswith(sw + "_")], key=lambda t: int(t.split("_")[1])) + [end]]
        ax.plot([r["lp"]["Ttot"] for r in R_], [r["lp"][yk] for r in R_], "-", color=LP, label="LP relaxation (lower bound)")
        ax.plot([r["int"]["Ttot"] for r in R_], [r["int"][yk] for r in R_], "o", color=LP, ms=7, mfc="white", mew=2,
                label=r"rounded vocabulary ($\varepsilon$-constraint)")
        Wr = [rc[t] for t in sorted([t for t in rc if t.startswith(wsw + "_")], key=lambda t: int(t.split("_")[1]))]
        ax.plot([r["int"]["Ttot"] for r in Wr], [r["int"][yk] for r in Wr], "s", mfc="none", mec=BPE, ms=8, mew=1.8,
                label="rounded vocabulary (weighted sum)")
        ax.set_xlabel(r"total cost $T_{\mathrm{tot}}$ (tokens / 1000 chars)"); ax.set_ylabel(ylab)
        ax.ticklabel_format(useOffset=False, style="plain")
    axs[0].legend(loc="upper right", fontsize=11.5)
    axs[0].set_title("(a) compression vs parity", loc="left"); axs[1].set_title("(b) compression vs morphology", loc="left")
    fig.tight_layout(w_pad=3)
    save(fig, name)


train_front(S10k, "s_train_headonly.png")
train_front(S10t, "s_train_saa.png")


# ---------------------------------------------------------------- 3. coverage (head-only 10k)
cov = json.load(open(f"{R}/full_10k/problem_stats.json"))["coverage"]
fig, ax = plt.subplots(figsize=(4.4, 3.5))
v = [100 * cov[l] for l in LANGS]
ax.bar(range(6), v, color=["#2a9d8f"] * 6, width=0.65)
for i, x in enumerate(v):
    ax.text(i, x + 1.5, f"{x:.0f}", ha="center", fontsize=12.5, color=INK)
ax.set_xticks(range(6)); ax.set_xticklabels(LANGS); ax.set_ylim(0, 100); ax.set_ylabel("text covered by trees (%)")
ax.grid(axis="x", visible=False)
save(fig, "s_coverage.png")


# ---------------------------------------------------------------- 4. held-out fronts (two panels)
def chain(prefix, S, sweep):
    b = {r["tag"]: r.get("tau", r.get("eps", 0)) for r in S["records"]}
    tags = ["compression"] + sorted([t for t in b if t.startswith(sweep + "_")], key=lambda t: -b[t]) + ["minmax" if sweep == "parity" else "min_morph"]
    return [prefix + t for t in tags if prefix + t in E]


def fronts(xf, yf, ax, sweep="parity", annotate=True, offs=None, aoffs=None):
    offs = offs or [(8, 6), (8, -14)]
    aoffs = aoffs or [(7, -4), (-6, -20)]
    head = chain("lp10k_", S10k, sweep)
    ax.plot([xf(E[n]) for n in head], [yf(E[n]) for n in head], "-", color=LPL, lw=2, marker=".", ms=6, label="head-only LP (10k)")
    main = chain("lp10t_", S10t, sweep)
    ax.plot([xf(E[n]) for n in main], [yf(E[n]) for n in main], "-o", color=LP, ms=6, label="SAA LP (10k+10k)", zorder=4)
    for pat, col, mk, lab in [("bpe_a", BPE, "D", r"BPE, temperature $\alpha$"), ("unigram_a", UNI, "^", r"Unigram, temperature $\alpha$")]:
        P = sorted([n for n in E if n.startswith(pat)], key=lambda n: float(n.split("_a")[1]))
        ax.plot([xf(E[n]) for n in P], [yf(E[n]) for n in P], "-" + mk, color=col, ms=7, lw=1.8, label=lab, zorder=3)
        if annotate and pat == "bpe_a":
            for n, ao in zip((P[0], P[-1]), aoffs):
                ax.annotate(r"$\alpha=%s$" % n.split("_a")[1].rstrip("0").rstrip("."), (xf(E[n]), yf(E[n])), xytext=ao,
                            textcoords="offset points", fontsize=12, color=INK2)
    ax.plot(xf(E["paritybpe_base"]), yf(E["paritybpe_base"]), "P", color=PAR, ms=13, label="Parity-aware BPE", zorder=5)
    ax.plot(xf(E["morphbpe"]), yf(E["morphbpe"]), "X", color=MOR, ms=12, label="MorphBPE", zorder=5)
    for n, lab, off in [(main[0], "compression anchor", offs[0]), (main[-1], "min-max anchor" if sweep == "parity" else "min-morph anchor", offs[1])]:
        ax.plot(xf(E[n]), yf(E[n]), "*", color=INK, ms=15, zorder=6)
        ax.annotate(lab, (xf(E[n]), yf(E[n])), xytext=off, textcoords="offset points", fontsize=12, color=INK)


fig, axs = plt.subplots(1, 2, figsize=(12.1, 4.55))
fronts(Tw, lambda r: r["T_max"], axs[0])
axs[0].set_xlabel("total cost (tokens / 1000 chars, data-weighted)"); axs[0].set_ylabel("worst-language cost")
axs[0].set_title("(a) held-out FineWeb text", loc="left")
fronts(flm, flx, axs[1])
axs[1].set_xlabel("mean tokens per FLORES+ sentence"); axs[1].set_ylabel("worst-language tokens per sentence")
axs[1].set_title("(b) FLORES+ devtest (parallel)", loc="left")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=12, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=3)
save(fig, "s_fronts_heldout.png")

# ---------------------------------------------------------------- 6. morphology (held-out F1)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
fronts(Tw, lambda r: r["ms_f1_mean"], ax, sweep="morph", offs=[(14, -6), (8, -20)], aoffs=[(-14, -22), (8, 2)])
ax.set_xlabel("total cost (tokens / 1000 chars, data-weighted)"); ax.set_ylabel("gold morpheme-boundary F1")
ax.legend(loc="upper center", ncol=2, fontsize=11.5, bbox_to_anchor=(0.5, 1.02))
ax.set_ylim(0.17, 0.66)
save(fig, "s_morph.png")


# ---------------------------------------------------------------- 7. LM bits per byte
LM = {}
for f in glob.glob(f"{R}/full/lm/*_s[0-9].json"):
    d = json.load(open(f))
    if d["args"].get("epochs", 1) >= 4:
        LM.setdefault(d["tokenizer"], []).append(d["fw_bpb_mean"])
show = [("morphbpe", "MorphBPE", MOR), ("unigram_a1.0", "Unigram (natural mix)", UNI), ("lp10k_compression", "head-only LP: compression", LPL),
        ("paritybpe_base", "Parity-aware BPE", PAR), ("bpe_a1.0", "BPE (natural mix)", BPE), ("bpe_a0.0", "BPE (uniform mix)", "#b8481c"),
        ("lp10t_compression", "SAA LP: compression", LP), ("lp10t_parity_5", "SAA LP: parity point", LP)]
fig, ax = plt.subplots(figsize=(7.6, 4.5))
y = np.arange(len(show))[::-1]
m = [np.mean(LM[k]) for k, _, _ in show]; sd = [np.std(LM[k]) for k, _, _ in show]
ax.barh(y, m, xerr=sd, color=[c for _, _, c in show], height=0.66, error_kw={"ecolor": INK2, "lw": 1.4, "capsize": 3})
for yy, mm in zip(y, m):
    ax.text(1.0845, yy, f"{mm:.3f}", va="center", ha="left", fontsize=12, color="white", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([n for _, n, _ in show]); ax.set_xlim(1.08, 1.215)
ax.set_xlabel("bits per byte on held-out FineWeb (lower is better)"); ax.grid(axis="y", visible=False)
save(fig, "s_lm.png")
json.dump({k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)} for k, v in LM.items()}, open(OUT + "lm_summary.json", "w"), indent=1)

# ---------------------------------------------------------------- 8. hypervolume
fig, ax = plt.subplots(figsize=(4.4, 3.55))
labs = ["head-only LP", "SAA LP 5k+5k", "SAA LP 10k+10k", "BPE family", "all baselines"]
vals = [0.082, 0.367, 0.749, 0.783, 0.925]
y = np.arange(5)[::-1]
ax.barh(y, vals, color=[LPL, LPL, LP, BPE, GREY], height=0.62)
for yy, v_ in zip(y, vals):
    ax.text(v_ + 0.02, yy, f"{v_:.2f}", va="center", fontsize=12.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, 1.12); ax.set_xlabel("normalised hypervolume")
ax.grid(axis="y", visible=False)
save(fig, "s_hv.png")

# ---------------------------------------------------------------- backup: shadow prices, scaling, solvers
rc = {r["tag"]: r for r in S10t["records"] if "lp" in r}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
for ax, sw, key, bkey, lab in [(axs[0], "parity", "t", "tau", r"parity bound $\tau$"), (axs[1], "morph", "Mtot", "eps", r"morphology bound $\varepsilon$")]:
    Rr = sorted([r for t, r in rc.items() if t.startswith(sw + "_")], key=lambda r: r[bkey])
    ax.plot([r[bkey] for r in Rr], [-r["rc"][key] for r in Rr], "-o", color=LP, ms=7)
    ax.set_xlabel(lab); ax.set_ylabel(r"$-\partial T^{*}_{\mathrm{tot}} / \partial(\mathrm{bound})$")
axs[0].set_title("(a) price of parity", loc="left"); axs[1].set_title("(b) price of morphology", loc="left")
fig.tight_layout(w_pad=3)
save(fig, "s_shadow.png")

runs = [("full", 0.360398, 65, None), ("full_10k", 0.769129, 312, 1690), ("full_20k", 1.616225, 1171, 8708), ("full_40k", 3.37125, 3323, None)]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot([r[1] for r in runs], [r[2] for r in runs], "-o", color=LP, ms=8, label="compression anchor (cold)")
mm = [(r[1], r[3]) for r in runs if r[3]]
ax.plot(*zip(*mm), "-s", color=BPE, ms=8, label="min-max anchor (cold)")
ax.plot([0.769129, 1.616225], [119, 361], "-^", color="#2a9d8f", ms=8, label="sweep point (warm, median)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.36, 0.77, 1.6, 3.4]); ax.set_xticklabels(["0.36", "0.77", "1.6", "3.4"]); ax.minorticks_off()
ax.set_yticks([100, 1000, 10000]); ax.set_yticklabels(["100", "1 000", "10 000"])
ax.set_xlabel("LP node variables (millions)"); ax.set_ylabel("solve time (s)"); ax.legend(fontsize=11.5, loc="upper left")
save(fig, "s_scaling.png")

fig, ax = plt.subplots(figsize=(5.2, 4.0))
labs = ["dual simplex", "parallel dual", "HiPO (interior point)", "IPX (interior point)", "PDLP (first order)"]
vals = [50, 42, 61, 73, 87]
y = np.arange(5)[::-1]
ax.barh(y, vals, color=[LP, LP, GREY, GREY, GREY], height=0.62)
for yy, v_ in zip(y, vals):
    ax.text(v_ + 1.5, yy, f"{v_} s", va="center", fontsize=12.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, 100); ax.set_xlabel("solve time, 5k problem (s)")
ax.grid(axis="y", visible=False)
save(fig, "s_solvers.png")
