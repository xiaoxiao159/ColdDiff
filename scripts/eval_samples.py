"""
Evaluate sampling outputs saved by sg_sample_fid: <ckpt>_<i>_pt.png / _gt.png.

Metrics per directory:
  - PSNR / SSIM (on [0,255])
  - Low-frequency error: mean |B_sigma(pred - gt)|  (FDO signature metric;
    the low-frequency anchor's gains show up here, not necessarily in PSNR)
  - LD reference: mean |B_sigma(ld - gt)| matched by dataset index (the
    de-noising target's low-frequency error baseline)

Usage:
  python scripts/eval_samples.py \
    --dirs FDO-cosine-150k=LoDoPaB_C_cosine/image_samples/LoDoPaB/images_fid \
           oldloss-cosine-600k=LoDoPaB_10_cosine/image_samples/LoDoPaB/images_fid \
    --fig3_out docs/figures/fig3_lowfreq_error_profile.png
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def to255(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def eval_dir(d, sigma, ld_paths, max_n=0, lowfreq_profiles=None):
    pts = sorted(glob.glob(os.path.join(d, "*_pt.png")),
                 key=lambda p: int(re.search(r"_(\d+)_pt\.png$", p).group(1)))
    if max_n:
        pts = pts[:max_n]
    psnrs, ssims, lf_pred, lf_ld = [], [], [], []
    for pp in pts:
        m = re.search(r"_(\d+)_pt\.png$", pp)
        idx = int(m.group(1))
        gp = pp.replace("_pt.png", "_gt.png")
        if not os.path.exists(gp):
            continue
        pred, gt = to255(pp), to255(gp)
        psnrs.append(psnr(pred, gt))
        if ssim_fn is not None:
            ssims.append(ssim_fn(gt, pred, data_range=255))
        d_lp = gaussian_filter(pred - gt, sigma)
        lf_pred.append(np.abs(d_lp).mean())
        if lowfreq_profiles is not None:
            lowfreq_profiles.append(np.abs(d_lp).mean(axis=0))
        if ld_paths is not None and idx < len(ld_paths):
            ld = to255(ld_paths[idx])
            lf_ld.append(np.abs(gaussian_filter(ld - gt, sigma)).mean())
    return {
        "n": len(psnrs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "lf_pred": float(np.mean(lf_pred)),
        "lf_ld": float(np.mean(lf_ld)) if lf_ld else float("nan"),
    }, (lowfreq_profiles if lowfreq_profiles is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="name=path entries")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--test_root", default="/home/zd/LoDoPaB_test",
                    help="for the LD reference curve (sorted *_ld.png order "
                         "matches dataset index)")
    ap.add_argument("--max_n", type=int, default=0)
    ap.add_argument("--fig3_out", default=None)
    args = ap.parse_args()

    ld_paths = sorted(glob.glob(os.path.join(args.test_root, "*", "*_ld.png")))

    print(f"{'name':<28}{'n':>6}{'PSNR':>9}{'SSIM':>9}"
          f"{'|LP(pred-gt)|':>15}{'|LP(ld-gt)|':>13}")
    profiles_first = None
    for entry in args.dirs:
        name, d = entry.split("=", 1)
        profiles = [] if (plt is not None and args.fig3_out
                          and profiles_first is None and len(args.dirs) > 1) \
                   else None
        r, profiles = eval_dir(d, args.sigma, ld_paths, args.max_n, profiles)
        if profiles:
            profiles_first = (name, profiles)
        print(f"{name:<28}{r['n']:>6}{r['psnr']:>9.3f}{r['ssim']:>9.4f}"
              f"{r['lf_pred']:>15.4f}{r['lf_ld']:>13.4f}")

    # Figure 3: low-frequency error profile, pred vs LD reference
    if args.fig3_out and plt is not None and profiles_first:
        name, profs = profiles_first
        ld_prof = []
        # recompute LD profile on the same subset for a fair curve
        pts = sorted(glob.glob(os.path.join(
            dict(e.split("=", 1) for e in args.dirs)[name], "*_pt.png")),
            key=lambda p: int(re.search(r"_(\d+)_pt\.png$", p).group(1)))
        for pp in pts[:len(profs)]:
            idx = int(re.search(r"_(\d+)_pt\.png$", pp).group(1))
            gp = pp.replace("_pt.png", "_gt.png")
            if idx < len(ld_paths) and os.path.exists(gp):
                ld = to255(ld_paths[idx])
                gt = to255(gp)
                ld_prof.append(np.abs(gaussian_filter(ld - gt, args.sigma))
                               .mean(axis=0))
        n = min(len(profs), len(ld_prof))
        plt.figure(figsize=(6.0, 4.0))
        plt.plot(np.mean(profs[:n], axis=0), label="Pred - ND (FDO output)")
        plt.plot(np.mean(ld_prof[:n], axis=0), label="LD - ND (input)")
        plt.xlabel("Low-frequency spatial profile (row, Gaussian-smoothed)")
        plt.ylabel("Mean absolute error (grey levels, 0-255)")
        plt.title(f"Low-frequency error profile (sigma={args.sigma})")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(args.fig3_out, dpi=200)
        plt.close()
        print(f"[fig3] saved -> {args.fig3_out}")


if __name__ == "__main__":
    main()
