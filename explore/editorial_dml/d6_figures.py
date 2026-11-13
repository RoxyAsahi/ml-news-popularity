"""D6: figures for the editorial-lever analysis (reads the CSVs in out/).

fig_levers.png         raw association vs the predictive model's what-if vs the adjusted (double ML) estimate
fig_dose_response.png  effect by bin of images, videos, body length and internal links
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
RAW, GBDT, DML = "#eb6834", "#1baf7a", "#2a78d6"   # categorical slots 2, 3, 1 (validated all-pairs)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
L = pd.read_csv(C.OUT / "d2_ladder.csv"); N = pd.read_csv(C.OUT / "d1_naive_levers.csv").set_index("lever")
J = L[(L.spec == "title") & (L["mode"] == "joint")].set_index("lever")
S1 = L[(L.spec == "title") & (L["mode"] == "single")].set_index("lever")
R0 = L[L.spec == "none"].set_index("lever")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [6, 4, 6], "wspace": 1.05})
panels = [(["imgs", "videos", "length", "title_len", "links_out", "links_self"], "Article: doubling (1 + count)", True),
          (["weekend", "vol_same", "vol_other"], "Day: weekend; doubling volume", True),
          (list(C.STYLE), "Wording: +1 standard deviation", False)]
for ax, (names, title, has_gbdt) in zip(axs, panels):
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8)
    for yi, l in zip(y, names):
        est = S1 if l == "weekend" else J
        ax.plot([est.pct_lo[l], est.pct_hi[l]], [yi, yi], color=DML, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(R0.pct[l], yi + 0.22, "o", color=RAW, ms=7, mec=SURF, mew=1.2, zorder=3)
        if has_gbdt:
            ax.plot(N.gbdt_whatif_pct[l], yi + 0.22, "s", color=GBDT, ms=6.5, mec=SURF, mew=1.2, zorder=3)
        ax.plot(est.pct[l], yi, "D", color=DML, ms=7, mec=SURF, mew=1.2, zorder=4)
        ax.annotate(f"{est.pct[l]:+.1f}%", (est.pct_hi[l], yi), xytext=(5, -3), textcoords="offset points", color=INK, fontsize=8)
        if l == "weekend":
            ax.plot(J.pct[l], yi - 0.25, "D", mfc=SURF, mec=DML, ms=6.5, mew=1.5, zorder=4)
            ax.annotate("at fixed volume", (J.pct[l], yi - 0.25), xytext=(7, -3), textcoords="offset points", color=MUTED, fontsize=7.5)
    ax.set_yticks(y, [(C.LEVERS.get(l) or C.STYLE[l])[0] for l in names])
    ax.set_ylim(-0.6, len(names) - 0.4); ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK); ax.set_xlabel("change in shares (%)")
    lo, hi = ax.get_xlim(); ax.set_xlim(lo - 0.03 * (hi - lo), hi + 0.16 * (hi - lo))
h = [plt.Line2D([], [], marker="o", color=RAW, ls="", ms=7, label="raw association"),
     plt.Line2D([], [], marker="s", color=GBDT, ls="", ms=6.5, label="boosted-tree model's what-if"),
     plt.Line2D([], [], marker="D", color=DML, ls="-", lw=2, ms=7, label="double ML, topic + headline held fixed (95% CI)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09), labelcolor=INK)
fig.savefig(C.OUT / "fig_levers.png", dpi=200, bbox_inches="tight"); plt.close(fig)

D = pd.read_csv(C.OUT / "d3_dose_response.csv", dtype={"bin": str}); D = D[D["mode"] == "joint"]
show = [("imgs", "Images"), ("videos", "Videos"), ("length", "Body length (100 words)"), ("links_self", "Links to Mashable articles")]
fig, axs = plt.subplots(1, 4, figsize=(13, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, (lv, title) in zip(axs, show):
    d = D[D.lever == lv].reset_index(drop=True); x = np.arange(len(d))
    ax.axhline(0, color=MUTED, lw=0.8)
    for i, r in d.iterrows():
        if r.reference:
            ax.plot(i, 0, "D", mfc=SURF, mec=DML, ms=7, mew=1.5); ax.annotate("reference", (i, 0), xytext=(0, 8), textcoords="offset points", ha="center", color=MUTED, fontsize=7.5)
        else:
            ax.plot([i, i], [r.pct_lo, r.pct_hi], color=DML, lw=2, solid_capstyle="round"); ax.plot(i, r.pct, "D", color=DML, ms=7, mec=SURF, mew=1.2)
            ax.annotate(f"{r.pct:+.0f}%", (i, r.pct_hi if r.pct > 0 else r.pct_lo), xytext=(0, 5 if r.pct > 0 else -12), textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, [b.replace("<250", "<2.5").replace("250-449", "2.5-4.5").replace("450-749", "4.5-7.5").replace("750-1199", "7.5-12").replace("1200+", "12+") for b in d.bin] if lv == "length" else d.bin); ax.set_xlim(-0.6, len(d) - 0.4); ax.tick_params(length=0); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
axs[0].set_ylim(-27, 18); axs[0].set_ylabel("change in shares (%)"); axs[0].tick_params(axis="y", colors=MUTED)
fig.savefig(C.OUT / "fig_dose_response.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures written")
