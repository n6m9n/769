import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/Users/naman/Desktop/769/motok/report/figures/"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LP, BPE, TEAL, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "figure.facecolor": BG, "axes.facecolor": BG})

def bars(fname, labels, vals, colors, title, fmt, horizontal, xlim=None, size=(5.2, 4.2)):
    fig, ax = plt.subplots(figsize=size)
    if horizontal:
        y = range(len(labels))[::-1]
        ax.barh(list(y), vals, color=colors, height=0.62)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for yy, v in zip(y, vals):
            ax.text(v + (xlim or max(vals)) * 0.015, yy, fmt.format(v), va="center", fontsize=11, color=INK)
        ax.set_xlim(0, (xlim or max(vals)) * 1.15)
    else:
        x = range(len(labels))
        ax.bar(list(x), vals, color=colors, width=0.62)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for xx, v in zip(x, vals):
            ax.text(xx, v + (xlim or max(vals)) * 0.015, fmt.format(v), ha="center", fontsize=11, color=INK)
        ax.set_ylim(0, xlim or max(vals) * 1.15)
    ax.set_title(title, fontsize=12, color="#1B1F3B", loc="left")
    fig.savefig(OUT + fname); plt.close(fig)

bars("fig_coverage.png", ["en", "hi", "mr", "ta", "ru", "hu"], [87.3, 89.2, 76.5, 66.1, 72.2, 72.2], [TEAL] * 6,
     "Text covered by trees (%), head-only, 10k trees / language", "{:.0f}", False, xlim=100)
bars("fig_hv.png", ["head-only LP 10k", "SAA LP 5k+5k", "SAA LP 10k+10k", "BPE α-sweep", "all 13 baselines"],
     [0.082, 0.367, 0.749, 0.783, 0.925], ["#8fb4e6", "#8fb4e6", LP, BPE, GREY], "Hypervolume, FineWeb (higher = better)", "{:.2f}", True, xlim=1.0)
bars("fig_solvers.png", ["dual simplex", "parallel dual", "HiPO (IPM)", "IPX (IPM)", "PDLP"], [50, 42, 61, 73, 87],
     [LP, LP, GREY, GREY, GREY], "Compression anchor, 5k problem (seconds)", "{:.0f}", True, xlim=87)
print("ok")
