"""
FDO (Frequency-Decoupled Objective) supporting measurements.

1) Radial power spectra of LD vs ND (Figure 1) + spectral-crossover
   frequency f_c -> principled Gaussian low-pass sigma for the FDO
   low-frequency anchor (half-power at the crossover).
2) Low-frequency bias statistics  c = mean[ B_sigma(LD) - B_sigma(ND) ]
   in the [0,1] image domain (reported; *2 for the [-1,1] training domain),
   validating/refreshing config `lf_bias`.
3) Figure 3 helper: spectral-difference curve of predictions vs ND,
   compared against the LD-ND baseline curve (run after sampling produces
   denoised outputs).

Usage:
  python scripts/spectral_analysis.py                       # measurements + Figure 1
  python scripts/spectral_analysis.py --pred_dir DIR        # + Figure 3 from outputs
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def radial_power_spectrum(img):
    """Azimuthally averaged power spectrum. img: [H,W] float, returns
    (freqs cycles/pixel, mean power per annulus)."""
    f = np.fft.rfft2(img - img.mean())
    ps = np.abs(f) ** 2
    h, w = ps.shape
    fy = np.fft.fftfreq(img.shape[0])[:, None]      # cycles/pixel
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    r_max = 0.5 * np.sqrt(2)                        # corner (diag Nyquist)
    bins = np.linspace(0.0, r_max, 65)
    idx = np.digitize(r.ravel(), bins) - 1
    freqs, power = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 4:
            continue
        freqs.append(r.ravel()[m].mean())
        power.append(ps.ravel()[m].mean())
    return np.array(freqs), np.array(power)


def spectral_crossover(freqs, r_excess, thresh=0.1, sustain=3):
    """First frequency where (PS_LD - PS_FD)/PS_FD exceeds `thresh`
    for `sustain` consecutive bins, searched above the DC plateau."""
    for i in range(2, len(r_excess) - sustain):
        if np.all(r_excess[i:i + sustain] > thresh):
            return freqs[i]
    return None


def sigma_from_crossover(f_c):
    """Gaussian sigma whose frequency response H(f)=exp(-2 pi^2 sigma^2 f^2)
    equals 0.5 (half power) at the crossover frequency f_c [cycles/pixel]."""
    return float(np.sqrt(np.log(2.0)) / (np.sqrt(2.0) * np.pi * f_c))


def load_pairs(dataroot, n_pairs, seed):
    lds = sorted(glob.glob(os.path.join(dataroot, "*", "*_ld.png")))
    rng = random.Random(seed)
    rng.shuffle(lds)
    pairs = []
    for p in lds:
        q = p.replace("_ld.png", "_fd.png")
        if os.path.exists(q):
            pairs.append((p, q))
        if len(pairs) >= n_pairs:
            break
    assert pairs, f"no ld/fd pairs under {dataroot}"
    return pairs


def to_arr(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def low_freq_bias(pairs, sigma):
    """c = mean[B_sigma(LD) - B_sigma(ND)] over pairs, in [0,1] domain."""
    diffs = []
    for lp, fp in pairs:
        ld = to_arr(lp)
        nd = to_arr(fp)
        diffs.append(gaussian_filter(ld, sigma) - gaussian_filter(nd, sigma))
    v = np.stack(diffs)
    return float(v.mean()), float(np.abs(v).mean()), float(v.std())


def fig1(freqs_ld, ps_ld, freqs_nd, ps_nd, f_c, sigma, out):
    plt.figure(figsize=(6.4, 4.4))
    plt.loglog(freqs_nd, ps_nd, label="NDCT (ground truth)", lw=1.8)
    plt.loglog(freqs_ld, ps_ld, label="LDCT (observation)", lw=1.8)
    plt.axvline(f_c, color="crimson", ls="--", lw=1.2,
                label=f"spectral crossover $f_c$={f_c:.4f} cyc/px\n"
                      f"$\\rightarrow\\ \\sigma$={sigma:.2f} px (half-power)")
    plt.xlabel("Spatial frequency (cycles/pixel)")
    plt.ylabel("Radially averaged power")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[fig1] saved -> {out}")


def fig3(pred_dir, gt_dir, sigma, out, n_max=50, seed=0):
    """Spectral difference |Pred-ND| vs |LD-ND| curves.
    pred_dir: generated *_ld.png-like outputs; matched to GT by basename
    with suffix replaced to _fd.png (same convention as sampling eval)."""
    if plt is None:
        raise RuntimeError("matplotlib unavailable")
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.png")))[:n_max]
    assert preds, f"no png under {pred_dir}"
    rng = random.Random(seed)
    rng.shuffle(preds)
    d_pred, d_ld = [], []
    for pp in preds:
        base = os.path.basename(pp)
        gp = os.path.join(gt_dir, base.replace("_ld.png", "_fd.png"))
        lp = os.path.join(gt_dir, base)
        if not (os.path.exists(gp) and os.path.exists(lp)):
            continue
        pred, nd, ld = to_arr(pp), to_arr(gp), to_arr(lp)
        d_pred.append(gaussian_filter(pred - nd, sigma))
        d_ld.append(gaussian_filter(ld - nd, sigma))
    dp, dl = np.stack(d_pred), np.stack(d_ld)
    n = min(len(dp), len(dl), 50)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(np.abs(dp[:n]).mean(axis=(0, 2)) * 255, label="|Pred - ND| (ours)")
    ax.plot(np.abs(dl[:n]).mean(axis=(0, 2)) * 255, label="|LD - ND| (baseline input)")
    ax.set_xlabel("Low-frequency row index (blur domain, per-row profile)")
    ax.set_ylabel("Mean abs low-frequency error (HU-ish, x255/255 scale)")
    ax.set_title(f"Low-frequency error profile (sigma={sigma})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig3] saved -> {out}  "
          f"mean|LP(Pred-ND)|={np.abs(dp).mean():.5f}  "
          f"mean|LP(LD-ND)|={np.abs(dl).mean():.5f}  (n={len(dp)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/zd/LoDoPaB_train")
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="sigma used for bias statistics / fig3 smoothing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="docs/figures")
    ap.add_argument("--pred_dir", default=None,
                    help="sampling outputs dir -> also produce Figure 3")
    ap.add_argument("--gt_dir", default="/home/zd/LoDoPaB_test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = load_pairs(args.dataroot, args.n_pairs, args.seed)
    print(f"loaded {len(pairs)} ld/fd pairs from {args.dataroot}")

    # ---- 1) radial spectra + crossover ----
    ps_ld_all, ps_nd_all, fr = [], [], None
    for lp, fp in pairs:
        fl, pl = radial_power_spectrum(to_arr(lp))
        fn, pn = radial_power_spectrum(to_arr(fp))
        fr = fl
        ps_ld_all.append(pl)
        ps_nd_all.append(pn)
    ps_ld = np.mean(ps_ld_all, axis=0)
    ps_nd = np.mean(ps_nd_all, axis=0)
    r_excess = (ps_ld - ps_nd) / ps_nd
    f_c = spectral_crossover(fr, r_excess)
    sigma_rec = sigma_from_crossover(f_c)
    print(f"spectral crossover f_c = {f_c:.4f} cyc/px  "
          f"-> recommended sigma (half-power) = {sigma_rec:.2f} px "
          f"(current config lf_sigma = {args.sigma})")

    # ---- 2) low-frequency bias at config sigma ----
    c01, c01_abs, c01_std = low_freq_bias(pairs[:100], args.sigma)
    print(f"LF bias @sigma={args.sigma}: mean={c01:+.5f}  "
          f"abs-mean={c01_abs:.5f}  std={c01_std:.5f}  "
          f"(report in [0,1]; x2 for [-1,1] training domain = {2*c01:+.5f})")

    # ---- Figure 1 ----
    if plt is not None:
        fig1(fr, ps_ld, fr, ps_nd, f_c, sigma_rec,
             os.path.join(args.out_dir, "fig1_radial_power_spectrum.png"))

    # ---- Figure 3 (optional, after sampling) ----
    if args.pred_dir:
        fig3(args.pred_dir, args.gt_dir, args.sigma,
             os.path.join(args.out_dir, "fig3_lowfreq_error_profile.png"))

    # persist measurements for the paper
    meas = {
        "n_pairs": len(pairs),
        "f_c_cyc_per_px": round(float(f_c), 5),
        "recommended_sigma_halfpower": round(sigma_rec, 3),
        "config_sigma": args.sigma,
        "lf_bias_mean_01": round(c01, 5),
        "lf_bias_absmean_01": round(c01_abs, 5),
        "lf_bias_std_01": round(c01_std, 5),
        "lf_bias_mean_pm1": round(2 * c01, 5),
    }
    out_json = os.path.join(args.out_dir, "fdo_measurements.json")
    with open(out_json, "w") as f:
        json.dump(meas, f, indent=2)
    print(f"measurements -> {out_json}")


if __name__ == "__main__":
    main()
