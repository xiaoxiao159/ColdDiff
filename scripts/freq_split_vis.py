"""
Visualize the LD/ND frequency-band split used by the FDO low-frequency anchor:
for one test slice, show original / Gaussian low-pass / high-pass residual,
for both LDCT (observation) and NDCT (ground truth).

Usage:
  python scripts/freq_split_vis.py [--sigma 3.0] [--out docs/figures/fig_freq_split.png]
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def to_arr(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", default="/home/zd/LoDoPaB_test")
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="same as config cold_diffusion.lf_sigma")
    ap.add_argument("--case", default=None, help="case dir name; default first")
    ap.add_argument("--idx", type=int, default=0, help="slice index within case")
    ap.add_argument("--out", default="docs/figures/fig_freq_split.png")
    args = ap.parse_args()

    # match the eval convention: sorted *_ld.png order == dataset index
    lds = sorted(glob.glob(os.path.join(args.test_root, "*", "*_ld.png")))
    lp_path = lds[args.idx]
    fd_path = lp_path.replace("_ld.png", "_fd.png")
    print(f"slice: {os.path.relpath(lp_path, args.test_root)}  sigma={args.sigma}")

    ld, nd = to_arr(lp_path), to_arr(fd_path)

    rows, names = [], []
    for tag, img in (("LDCT (input)", ld), ("NDCT (ground truth)", nd)):
        low = gaussian_filter(img, args.sigma)
        high = img - low
        rows.append((img, low, high))
        names.append(tag)

    if plt is None:
        raise RuntimeError("matplotlib unavailable")

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.6))
    vmax = 1.0
    for r, (row, tag) in enumerate(zip(rows, names)):
        img, low, high = row
        # high-pass is near-zero-mean and small amplitude: stretch for display
        hmax = np.percentile(np.abs(high), 99.5)
        panels = [
            dict(data=img, vmin=0, vmax=1.0, cmap="gray",
                 title=f"{tag} - original"),
            dict(data=low, vmin=0, vmax=1.0, cmap="gray",
                 title=f"low-pass B_sigma(img), sigma={args.sigma}"),
            dict(data=high, vmin=-hmax, vmax=hmax, cmap="gray",
                 title=f"high-pass img-B_sigma(img) (contrast x{hmax:.4f})"),
        ]
        for c, p in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(p["data"], vmin=p["vmin"], vmax=p["vmax"],
                           cmap=p["cmap"])
            ax.set_title(p["title"], fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Frequency-band split at the FDO anchor scale "
                 "(low band: reliable in LDCT; high band: noise-dominated in LDCT)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
