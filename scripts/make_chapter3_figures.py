"""
Chapter-3 thesis figures for FBDG-ColdDiff (docs/第三章_方法.md).

Generates four figures into docs/figures/:
  fig3_1_framework.png   overall framework schematic (3 panels)
  fig3_2_schedules.png   alpha_bar schedules + per-step decrement bars
  fig3_3_spectrum.png    (a) mean radial power spectra LD vs ND + crossover;
                         (b) per-image crossover f_c histogram (the 3.5x spread)
  fig3_4_fbdg_module.png FBDG module schematic
  fig3_5_inference.png   1-D illustrative inference trajectory on the
                         degradation manifold (values from the walkthrough table)

Usage: python scripts/make_chapter3_figures.py [--skip-spectrum]
Palette: dataviz reference categorical slots (validated), print-friendly.
"""
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from runners.schedules import get_cold_alpha_schedule  # noqa: E402
import spectral_analysis as sa  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
DPI = 300

# palette (light mode, dataviz reference instance)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTE = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#d9d8d4"
BOX_FC, BOX_EC = "#ffffff", "#52514e"
ACC_FILL = "#eef3fa"   # light blue tint for emphasis boxes

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.8,
})


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def box(ax, x, y, w, h, text="", fc=BOX_FC, ec=BOX_EC, fs=8.5, lw=1.1,
        text_color=INK, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        fc=fc, ec=ec, lw=lw, ls=ls, mutation_scale=1))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=text_color, weight=weight, linespacing=1.35)


def arrow(ax, p1, p2, color=MUTE, lw=1.3, ls="-", shrink=2, style="-|>",
          cs=None, zorder=3):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, color=color, lw=lw, linestyle=ls,
        shrinkA=shrink, shrinkB=shrink, mutation_scale=9,
        connectionstyle=cs or "arc3,rad=0", zorder=zorder))


# --------------------------------------------------------------------------
# Fig 3-1 framework schematic
# --------------------------------------------------------------------------
def fig_framework(out):
    fig = plt.figure(figsize=(11.5, 4.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.6); ax.set_ylim(0, 4.6)
    ax.axis("off")

    # panel titles
    for x, t in ((0.15, "(a) Forward degradation (training only)"),
                 (4.45, "(b) Two-stage conditional U-Net  $f_\\theta$"),
                 (8.95, "(c) One iteration, repeated $T$ times")):
        ax.text(x, 4.38, t, fontsize=10, color=INK, weight="bold")

    # ---- (a) forward degradation ----
    box(ax, 0.25, 3.05, 1.25, 0.7, "NDCT\n$\\mathbf{x}_{ND}$", ec=C1, lw=1.4)
    box(ax, 0.25, 1.35, 1.25, 0.7, "LDCT\n$\\mathbf{x}_{LD}$", ec=C2, lw=1.4)
    box(ax, 2.05, 2.2, 2.0, 0.72,
        "$\\mathbf{x}_t=\\bar{\\alpha}_t\\,\\mathbf{x}_{ND}"
        "+(1-\\bar{\\alpha}_t)\\,\\mathbf{x}_{LD}$", fs=8)
    arrow(ax, (1.5, 3.3), (2.1, 2.72), color=C1)
    arrow(ax, (1.5, 1.8), (2.1, 2.42), color=C2)
    box(ax, 4.55 - 2.3, 0.62, 1.15, 0.62, "$\\mathbf{x}_t$\n(level $t$)",
        fc=ACC_FILL)
    arrow(ax, (3.1, 2.2), (3.35, 1.28))
    # alpha ramp strip
    n = 11
    for i in range(n):
        a = i / (n - 1)
        r, g, b = tuple((1 - a) * np.array([0xEB, 0x68, 0x34]) / 255
                        + a * np.array([0x2A, 0x78, 0xD6]) / 255)
        ax.add_patch(Rectangle((0.35 + i * 0.155, 0.16), 0.13, 0.26,
                               fc=(r, g, b), ec="none"))
    ax.text(0.35, 0.02, "$\\bar{\\alpha}_{10}=0$ (LDCT)", fontsize=7,
            color=INK2, ha="left")
    ax.text(2.03, 0.02, "$\\bar{\\alpha}_{0}=1$ (NDCT)",
            fontsize=7, color=INK2, ha="right")

    # ---- (b) network ----
    # input stack
    for i, (lab, col, ls) in enumerate((
            ("$\\mathbf{x}_{LD}$", C2, "-"),
            ("$\\mathbf{x}_t$", C1, "-"),
            ("$\\mathbf{r}_t$", C3, "--"))):
        y = 3.3 - i * 0.62
        ax.add_patch(Rectangle((4.55, y), 0.5, 0.5, fc="white", ec=col,
                               lw=1.2, ls=ls))
        ax.text(5.15, y + 0.25, lab, fontsize=8.5, va="center", color=INK)
    ax.text(4.62, 2.0, "3-ch input\n(stage 1: $\\mathbf{r}$ = 0)",
            fontsize=7, color=INK2, va="top")
    box(ax, 6.0, 1.7, 2.15, 1.75,
        "5-level U-Net\n32/64/96/128/160\n\nadditive time injection\n"
        "FBDG skips\nAttention @ 22$\\times$22", fc=ACC_FILL, ec=C1,
        lw=1.4, fs=7.6)
    box(ax, 6.0, 0.55, 2.15, 0.55, "time embedding  $t\\rightarrow\\mathbf{e}_t$",
        fs=7.6)
    arrow(ax, (7.05, 1.1), (7.05, 1.7))
    arrow(ax, (5.1, 2.35), (6.0, 2.45), color=INK2)
    box(ax, 8.2, 2.3, 0.7, 0.55, "$\\hat{\\mathbf{x}}_0$", fc=ACC_FILL)
    arrow(ax, (8.15, 2.57), (8.22, 2.57), color=C1)
    # x_t feeds the input stack
    arrow(ax, (3.42, 1.05), (4.5, 2.72), color=C1, ls=":", lw=1.1)

    # ---- (c) iteration ----
    x0, w = 8.95, 2.85
    box(ax, x0, 3.35, w, 0.62,
        "Stage 1 (coarse):  $\\hat{\\mathbf{x}}_0^{(1)}"
        "=f_\\theta([\\mathbf{x}_{LD},\\mathbf{x}_t,\\mathbf{0}],t)$",
        fs=7.4, fc=ACC_FILL)
    box(ax, x0, 2.45, w, 0.62,
        "Improved update:  $\\mathbf{x}_{t-1}=\\mathbf{x}_t"
        "+\\Delta\\bar{\\alpha}_t(\\hat{\\mathbf{x}}_0^{(1)}-\\mathbf{x}_{LD})$",
        fs=7.4)
    box(ax, x0, 1.55, w, 0.62,
        "Residual map:  $\\mathbf{r}_t=\\tanh((\\mathbf{x}_t"
        "-\\mathcal{D}_t(\\hat{\\mathbf{x}}_0^{(1)}))/\\sigma)$", fs=7.4)
    box(ax, x0, 0.65, w, 0.62,
        "Stage 2 (refine):  $\\hat{\\mathbf{x}}_0^{(2)}"
        "=f_\\theta([\\mathbf{x}_{LD},\\mathbf{x}_{t-1},\\mathbf{r}_t],t-1)$",
        fs=7.4, fc=ACC_FILL)
    arrow(ax, (x0 + 0.7, 3.35), (x0 + 0.7, 3.07))
    arrow(ax, (x0 + 0.7, 2.45), (x0 + 0.7, 2.17))
    arrow(ax, (x0 + 0.7, 1.55), (x0 + 0.7, 1.27))
    # loop back (routed outside the right edge of the boxes)
    arrow(ax, (x0 + w, 0.96), (x0 + w, 3.66), color=C2,
          cs="arc3,rad=-0.3")
    ax.text(x0 + w / 2, 0.48,
            "$t\\leftarrow t-1$;  inference: $T{=}10$ rounds, "
            "2 forwards each\ntraining: 1 random step;  "
            "$\\mathcal{L}=0.4\\mathcal{L}_1+0.6\\mathcal{L}_2$",
            fontsize=7.2, color=C2, ha="center", va="top", linespacing=1.6)
    # start marker
    arrow(ax, (8.55, 3.66), (8.95, 3.66), color=INK2)
    ax.text(8.52, 3.82, "$\\mathbf{x}_T=\\mathbf{x}_{LD}$", fontsize=7.4,
            ha="right", color=INK2)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[fig3-1] {out}")


# --------------------------------------------------------------------------
# Fig 3-2 schedules
# --------------------------------------------------------------------------
def fig_schedules(out):
    T = 10
    names = ["linear", "cosine", "sigmoid", "quad"]
    colors = [C1, C2, C3, C4]
    bars = {n: get_cold_alpha_schedule(n, T).numpy() for n in names}
    t = np.arange(0, T + 1)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for n, c in zip(names, colors):
        axa.plot(t, bars[n], "-o", color=c, lw=2, ms=4.5, label=n)
    axa.set_xlabel("timestep  $t$"); axa.set_ylabel("$\\bar{\\alpha}_t$")
    axa.set_xticks(range(T + 1)); axa.set_ylim(-0.04, 1.04)
    style_ax(axa)
    axa.legend(frameon=False, fontsize=8.5, loc="lower left")

    width = 0.2
    for k, (n, c) in enumerate(zip(names, colors)):
        d = -np.diff(bars[n])         # Δᾱ_t = ᾱ_{t-1} - ᾱ_t, t = 1..T
        axb.bar(np.arange(1, T + 1) + (k - 1.5) * width, d, width * 0.92,
                color=c, label=n, lw=0)
    axb.set_xlabel("timestep  $t$")
    axb.set_ylabel("$\\Delta\\bar{\\alpha}_t=\\bar{\\alpha}_{t-1}"
                   "-\\bar{\\alpha}_t$")
    axb.set_xticks(range(1, T + 1)); axb.set_ylim(0, 0.26)
    style_ax(axb)
    axb.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[fig3-2] {out}")


# --------------------------------------------------------------------------
# Fig 3-3 spectrum + per-image crossover spread
# --------------------------------------------------------------------------
def fig_spectrum(out, n_pairs=200):
    pairs = sa.load_pairs(os.path.join("/home/zd/LoDoPaB_train"), n_pairs, 0)
    freqs_acc = ps_ld_acc = ps_nd_acc = None
    f_c_list = []
    for lp, fp in pairs:
        ld, nd = sa.to_arr(lp), sa.to_arr(fp)
        f_ld, p_ld = sa.radial_power_spectrum(ld)
        f_nd, p_nd = sa.radial_power_spectrum(nd)
        if freqs_acc is None:
            freqs_acc = np.zeros_like(f_ld)
            ps_ld_acc = np.zeros_like(p_ld)
            ps_nd_acc = np.zeros_like(p_nd)
        freqs_acc += f_ld
        ps_ld_acc += p_ld
        ps_nd_acc += p_nd
        r_excess = (p_ld - p_nd) / p_nd
        f_c = sa.spectral_crossover(f_ld, r_excess)
        if f_c is not None:
            f_c_list.append(f_c)
    freqs = freqs_acc / len(pairs)
    ps_ld, ps_nd = ps_ld_acc / len(pairs), ps_nd_acc / len(pairs)
    f_c_mean = float(np.median(f_c_list))
    f_c_arr = np.array(f_c_list)
    p5, p95 = np.percentile(f_c_arr, 5), np.percentile(f_c_arr, 95)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    axa.loglog(freqs, ps_nd, color=C1, lw=2, label="NDCT (ground truth)")
    axa.loglog(freqs, ps_ld, color=C2, lw=2, label="LDCT (observation)")
    axa.axvline(0.1049, color="#c8102e", ls="--", lw=1.2,
                label="spectral crossover $f_c$=0.105 cyc/px")
    axa.set_xlabel("Spatial frequency (cycles/pixel)")
    axa.set_ylabel("Radially averaged power")
    style_ax(axa, grid_axis="both")
    axa.legend(frameon=False, fontsize=8, loc="lower left")

    axb.hist(f_c_arr, bins=24, color=C1, alpha=0.85, lw=0)
    axb.axvline(p5, color=INK2, ls="--", lw=1.2)
    axb.axvline(p95, color=INK2, ls="--", lw=1.2)
    axb.axvline(f_c_mean, color="#c8102e", ls="-", lw=1.4)
    ymax = axb.get_ylim()[1]
    axb.set_ylim(0, ymax * 1.3)          # headroom for the line labels
    axb.text(p5, ymax * 1.27, f"p5={p5:.3f}", fontsize=7.5, color=INK2,
             ha="center", va="top")
    axb.text(p95, ymax * 1.27, f"p95={p95:.3f}", fontsize=7.5, color=INK2,
             ha="center", va="top")
    axb.text(f_c_mean, ymax * 1.27, f"median={f_c_mean:.3f}",
             fontsize=7.5, color="#c8102e", ha="center", va="top")
    axb.set_xlabel("per-image spectral crossover  $f_c$ (cycles/pixel)")
    axb.set_ylabel("number of images")
    style_ax(axb)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[fig3-3] {out}   n={len(f_c_arr)}  "
          f"range [{f_c_arr.min():.4f},{f_c_arr.max():.4f}]  "
          f"p5={p5:.4f} p95={p95:.4f}")


# --------------------------------------------------------------------------
# Fig 3-4 FBDG module schematic
# --------------------------------------------------------------------------
def fig_fbdg(out):
    fig = plt.figure(figsize=(10.2, 5.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10.2); ax.set_ylim(0, 5)
    ax.axis("off")

    box(ax, 0.2, 2.2, 1.0, 0.7, "skip\n$\\mathbf{s}$", fc=ACC_FILL, ec=C1,
        lw=1.4)
    box(ax, 1.7, 2.2, 0.95, 0.7, "Haar DWT\n(fixed)")
    arrow(ax, (1.2, 2.55), (1.7, 2.55))

    # low-freq path (top)
    box(ax, 3.2, 3.6, 1.0, 0.62, "$\\mathbf{L}$ (low freq)", fs=8)
    box(ax, 4.9, 3.6, 2.1, 0.62,
        "$\\Delta\\mathbf{L}=\\mathrm{Conv}_{1\\times1}\\mathrm{SiLU}"
        "(\\mathrm{DWConv}(\\mathbf{L}))$\nzero-init", fs=7)
    box(ax, 7.8, 3.6, 0.75, 0.62, "$\\mathbf{L}'$", fc=ACC_FILL)
    arrow(ax, (2.65, 2.75), (3.2, 3.85), color=C1)
    arrow(ax, (4.2, 3.91), (4.9, 3.91), color=C1)
    arrow(ax, (7.0, 3.91), (7.8, 3.91), color=C1)
    ax.text(7.42, 4.05, "$+$", fontsize=11, color=INK, ha="center")

    # high-freq path (bottom)
    box(ax, 3.2, 0.55, 1.15, 0.62,
        "$\\mathbf{H}$ (high freq)\nLH/HL/HH", fs=7.4)
    arrow(ax, (2.65, 2.35), (3.2, 1.05), color=C2)

    # gate branch
    box(ax, 4.85, 1.75, 1.35, 0.55, "$\\mathrm{DWConv}(|\\mathbf{H}|)$",
        fs=7.4)
    box(ax, 4.85, 2.55, 1.35, 0.55, "$\\mathbf{W}_t\\,\\sigma(\\mathbf{e}_t)$"
        "\nzero-init", fs=7.4, ec=C3)
    box(ax, 6.7, 2.15, 0.6, 0.55, "$\\sigma$", fs=9)
    box(ax, 7.55, 2.15, 0.5, 0.55, "$\\mathbf{g}$", fc=ACC_FILL, ec=C2,
        lw=1.4)
    arrow(ax, (4.35, 1.0), (4.85, 1.95), color=C2)
    arrow(ax, (6.2, 2.02), (6.7, 2.35), color=C2)
    arrow(ax, (6.2, 2.82), (6.7, 2.5), color=C3)
    arrow(ax, (7.3, 2.42), (7.55, 2.42), color=C2)

    # correction branch
    box(ax, 4.85, 0.42, 1.35, 0.55,
        "$\\mathrm{DWConv}\\!\\rightarrow\\!\\mathrm{SiLU}\\!\\rightarrow\\!"
        "\\mathrm{Conv}_{1\\times1}$\nzero-init", fs=6.8, ec=C3)
    arrow(ax, (4.35, 0.86), (4.85, 0.7), color=C2)
    ax.text(6.62, 0.7, "$\\Delta\\mathbf{H}$", fontsize=8.5, color=INK,
            va="center")
    arrow(ax, (6.2, 0.7), (6.5, 0.7), color=C2)

    # combine
    box(ax, 8.3, 1.3, 1.65, 0.62,
        "$\\mathbf{H}'=\\mathbf{g}\\odot\\mathbf{H}"
        "+(1-\\mathbf{g})\\odot\\Delta\\mathbf{H}$", fs=7.4, fc=ACC_FILL)
    arrow(ax, (7.8, 2.42), (8.55, 1.95), color=C2)
    arrow(ax, (6.62, 0.7), (8.55, 1.32), color=C2)

    # idwt + output
    box(ax, 8.3, 3.0, 1.65, 0.55, "Haar IDWT (fixed)")
    box(ax, 8.75, 4.0, 0.75, 0.55, "$\\hat{\\mathbf{s}}$", fc=ACC_FILL,
        ec=C1, lw=1.4)
    arrow(ax, (8.55, 3.62), (8.55, 3.55), color=C1)   # L' into idwt
    arrow(ax, (9.12, 1.92), (9.12, 3.0), color=C2)
    arrow(ax, (9.12, 3.55), (9.12, 4.0), color=C1)

    ax.text(0.2, 0.85,
            "init:  $\\mathbf{g}=\\sigma(2)\\approx0.88$,  "
            "$\\Delta\\mathbf{H}=\\Delta\\mathbf{L}=\\mathbf{0}$"
            "  $\\Rightarrow$  near-identity\n"
            "same-shape in/out; skip copy only (encoder trunk keeps raw)",
            fontsize=7.6, color=INK2, va="top", linespacing=1.5)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[fig3-4] {out}")


# --------------------------------------------------------------------------
# Fig 3-5 inference trajectory (illustrative 1-D walkthrough)
# --------------------------------------------------------------------------
def fig_inference(out):
    # corrected walkthrough values (x_gt=0.6, x_LD=0.2), illustrative
    x_ld, x_nd = 0.2, 0.6
    ab = get_cold_alpha_schedule("linear", 10).numpy()     # index t = 0..10
    t = np.arange(1, 11)                                    # loop t = 10..1
    xhat1 = np.array([0.580, 0.582, 0.584, 0.586, 0.588,
                      0.589, 0.590, 0.591, 0.593, 0.595])   # stage-1 preds
    x = [0.2]
    for k in range(10):
        x.append(x[-1] + (ab[9 - k] - ab[10 - k]) * (xhat1[k] - x_ld))
    x = np.array(x)                                          # x_10..x_0
    ab_t = ab[10 - t]                                        # ᾱ at x_t
    xhat2 = xhat1 + 0.4 * (x_nd - xhat1)                     # stage-2 (illust.)

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    aa = np.linspace(0, 1, 50)
    ax.plot(aa, aa * x_nd + (1 - aa) * x_ld, ls="--", color=MUTE, lw=1.4,
            label="ideal trajectory  $\\mathcal{D}_t(\\mathbf{x}_{ND})$")
    ax.axhline(x_nd, color=C1, ls=":", lw=1.0, alpha=0.7)
    ax.text(0.995, x_nd + 0.004, "NDCT", color=C1, fontsize=8, ha="right")

    ax.plot(ab_t, x[:-1], "-", color=C2, lw=2,
            label="actual state  $\\mathbf{x}_t$", zorder=4)
    ax.plot(ab_t, x[:-1], "o", color=C2, ms=5, zorder=5)
    ax.plot([1.0], [x[-1]], "o", color=C2, ms=5, zorder=5)
    ax.plot(ab_t, xhat1, "o", mfc="white", mec=C2, ms=5.5, mew=1.4,
            label="stage-1 prediction  $\\hat{\\mathbf{x}}_0^{(1)}$",
            zorder=5)
    for k in range(10):     # link each state to its prediction
        ax.plot([ab_t[k], ab_t[k]], [x[:-1][k], xhat1[k]],
                color=C2, lw=0.7, ls=":", alpha=0.55, zorder=2)
    ax.plot([1.0], [xhat2[-1]], "*", color=C3, ms=15, mec=INK2, mew=0.4,
            label="final output (stage-2, $t{=}0$)", zorder=6)

    ax.annotate("$\\hat{\\mathbf{x}}_0$ almost final answer;\nupdate only "
                "moves $\\Delta\\bar{\\alpha}_t$",
                xy=(0.12, 0.582), xytext=(0.22, 0.50), fontsize=8,
                color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.9))
    ax.annotate("error $\\bar{\\alpha}_t(\\mathbf{x}_{ND}"
                "-\\hat{\\mathbf{x}}_0)\\rightarrow\\mathbf{r}_t$",
                xy=(0.5, 0.5922), xytext=(0.56, 0.44), fontsize=8,
                color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.9))
    ax.set_xlabel("degradation level  $\\bar{\\alpha}_t$   "
                  "(inference: $t=10\\rightarrow 0$, left to right)")
    ax.set_ylabel("pixel value (illustrative)")
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(0.15, 0.66)
    style_ax(ax, grid_axis="both")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[fig3-5] {out}  final x0={x[-1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-spectrum", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig_framework(os.path.join(OUT, "fig3_1_framework.png"))
    fig_schedules(os.path.join(OUT, "fig3_2_schedules.png"))
    if not args.skip_spectrum:
        fig_spectrum(os.path.join(OUT, "fig3_3_spectrum.png"))
    fig_fbdg(os.path.join(OUT, "fig3_4_fbdg_module.png"))
    fig_inference(os.path.join(OUT, "fig3_5_inference.png"))


if __name__ == "__main__":
    main()
