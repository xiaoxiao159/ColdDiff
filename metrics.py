#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 result 目录下 _pt(预测) 与 _gt(全剂量 GT) 图像对的客观指标:
    PSNR / SSIM / RMSE / LPIPS / VIF / CNR

- _pt 为失真(预测)图像, _gt 为参考(GT)图像
- 目录格式: result/<病例>/<切片>_gt.png 与 <切片>_pt.png, 按同路径同名配对
- 所有指标先转灰度后在 [0,255] 范围计算 (CT 为灰度图)
- 结果打印汇总, 并保存到同目录下的 metrics_results.csv

用法:
    python metrics.py                     # 使用默认目录 result/
    python metrics.py -d <目录>           # 指定目录
    python metrics.py --csv out.csv      # 指定输出 csv
可选依赖:
    pip install lpips        # 若安装了 torch+lpips 则计算 LPIPS, 否则跳过
"""

import argparse
import csv
import os
import re
import sys
import glob

import numpy as np
from scipy.signal import convolve2d
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as _skimage_ssim

IMG_EXTS = ('.png', '.jpt', '.jpeg', '.bmp', '.tif', '.tiff')


# ----------------------------------------------------------------------
# 基础指标 (纯 numpy)
# ----------------------------------------------------------------------
def psnr(ref, dist, data_range=255.0):
    mse = np.mean((ref - dist) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(data_range ** 2 / mse)


def rmse(ref, dist):
    return np.sqrt(np.mean((ref - dist) ** 2))


def nmse(ref, dist):
    """Normalized mean squared error: ||dist-ref||^2 / ||ref||^2"""
    return np.sum((dist - ref) ** 2) / (np.sum(ref ** 2) + 1e-12)


def ssim(ref, dist, data_range=255.0):
    return _skimage_ssim(ref, dist, data_range=data_range,
                         gaussian_weights=True, sigma=1.5,
                         use_sample_covariance=False)


# ----------------------------------------------------------------------
# VIF (pixel domain, Sheik & Bovik 2006)
# ----------------------------------------------------------------------
def vifp(ref, dist, sigma_nsq=2.0):
    eps = 1e-10
    num = 0.0
    den = 0.0
    r = ref.copy()
    d = dist.copy()
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        if scale > 1:
            r = r[::2, ::2]
            d = d[::2, ::2]
        m = (N - 1) // 2
        y, x = np.mgrid[-m: m + 1, -m: m + 1]
        h = np.exp(-((x ** 2 + y ** 2) / (2.0 * sd ** 2)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        h /= h.sum()

        mu1 = convolve2d(r, h, mode='valid')
        mu2 = convolve2d(d, h, mode='valid')
        sigma1_sq = convolve2d(r * r, h, mode='valid') - mu1 ** 2
        sigma2_sq = convolve2d(d * d, h, mode='valid') - mu2 ** 2
        sigma12 = convolve2d(r * d, h, mode='valid') - mu1 * mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0
        sv_sq[g < 0] = 0

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    return float(num / (den + eps))


# ----------------------------------------------------------------------
# CNR (Contrast-to-Noise Ratio, 对比噪声比)
# 用 _gt(GT) 的 Otsu 阈值划分目标区(体部)与背景区,
# 在 _pt(预测) 上计算: CNR = |mu_in - mu_out| / sqrt(sd_in^2 + sd_out^2)
# ----------------------------------------------------------------------
def cnr(ref, dist):
    from skimage.filters import threshold_otsu
    thr = threshold_otsu(ref)
    mask_in = ref > thr
    mask_out = ~mask_in
    if mask_in.sum() < 10 or mask_out.sum() < 10:
        return np.nan
    mu_in, mu_out = dist[mask_in].mean(), dist[mask_out].mean()
    sd_in, sd_out = dist[mask_in].std(), dist[mask_out].std()
    return float(abs(mu_in - mu_out)
                 / np.sqrt(sd_in ** 2 + sd_out ** 2 + 1e-12))


# ----------------------------------------------------------------------
# LPIPS (可选, 需要 torch + lpips)
# ----------------------------------------------------------------------
def make_lpips_fn(net='alex'):
    try:
        import torch
        import lpips as lpips_lib
        model = lpips_lib.LPIPS(net=net)
        torch.set_grad_enabled(False)

        def fn(ref, dist):
            # 输入: [0,255] float -> [-1,1] tensor
            def to_t(x):
                t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
                return t[None, None, :, :] if t.ndim == 2 else \
                    t[None].permute(0, 3, 1, 2)
            return float(model(to_t(ref), to_t(dist)))
        return fn
    except Exception as e:
        print('[警告] LPIPS 不可用 (需要 torch + lpips): %s' % e)
        return None


# ----------------------------------------------------------------------
# 数据读取与配对
# ----------------------------------------------------------------------
def to_gray(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.shape[2] == 3:
            img = rgb2gray(img) * 255.0
        else:
            img = img[:, :, 0]
    return img.astype(np.float64)


def collect_pairs(img_dir, prefix=None):
    """遍历 img_dir (含子目录), 将 <stem>_pt.png 与 <stem>_gt.png 按同路径配对.

    prefix 不为 None 时, 只保留 stem 以该前缀开头的图像对 (如 '600000').

    返回 [(key, pt_path, gt_path), ...], key 为 '相对路径/stem' (不含后缀).
    """
    pairs = []
    gt_files = {}
    pt_files = {}
    for root, _dirs, files in os.walk(img_dir):
        for f in sorted(files):
            name, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXTS:
                continue
            rel = os.path.relpath(root, img_dir)
            if name.endswith('_gt'):
                key = name[:-3] if rel == '.' else os.path.join(rel, name[:-3])
                gt_files[key] = os.path.join(root, f)
            elif name.endswith('_pt'):
                key = name[:-3] if rel == '.' else os.path.join(rel, name[:-3])
                pt_files[key] = os.path.join(root, f)
    for key in sorted(gt_files.keys()):
        if prefix is not None and not key.startswith(prefix):
            continue
        if key in pt_files:
            pairs.append((key, pt_files[key], gt_files[key]))
        else:
            print('[警告] 缺少 %s_pt, 跳过' % key)
    return pairs


def main():
    ap = argparse.ArgumentParser(description='_pt vs _gt 指标评估')
    default_dir = '/home/zd/baseline/LoDoPaB_BC/image_samples/LoDoPaB/images_fid'
    ap.add_argument('-d', '--dir', default=default_dir, help='图像目录')
    ap.add_argument('--csv', default=None, help='输出 csv 路径')
    ap.add_argument('--prefix', default=None,
                    help='只评估文件名以该前缀开头的图像对 (如 600000)')
    args = ap.parse_args()

    img_dir = os.path.abspath(args.dir)
    pairs = collect_pairs(img_dir, prefix=args.prefix)
    if not pairs:
        print('错误: 在 %s 中未找到 *_gt / *_pt 图像对' % img_dir)
        sys.exit(1)
    print('找到 %d 对图像 (目录: %s)' % (len(pairs), img_dir))

    lpips_fn = make_lpips_fn()

    metrics = ['PSNR', 'SSIM', 'RMSE', 'LPIPS', 'VIF', 'CNR']
    rows = []

    for i, (key, pt_path, gt_path) in enumerate(pairs, 1):
        pt = to_gray(imread(pt_path))
        gt = to_gray(imread(gt_path))
        if pt.shape != gt.shape:
            print('[警告] 尺寸不一致 %s: %s vs %s, 跳过'
                  % (key, pt.shape, gt.shape))
            continue

        row = {'name': key,
               'PSNR': psnr(gt, pt),
               'SSIM': ssim(gt, pt),
               'RMSE': rmse(gt, pt),
               'LPIPS': lpips_fn(gt, pt) if lpips_fn else np.nan,
               'VIF': vifp(gt, pt),
               'CNR': cnr(gt, pt)}
        rows.append(row)
        vals = '  '.join('%s=%.4f' % (m, row[m]) for m in metrics)
        print('[%d/%d] %s  %s' % (i, len(pairs), key, vals))

    # 汇总: 保留 4 位有效数字, 格式 mean+-std
    print('\n========== 汇总 (%d 对图像) ==========' % len(rows))
    summary = {}
    summary_line = []
    for m in metrics:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        summary[m] = float(np.mean(vals)) if vals else np.nan
        if vals:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            print('%-6s : %s +-%s' % (m, '%.4g' % mean, '%.4g' % std))
            summary_line.append('%s=%.4g+-%.4g' % (m.lower(), mean, std))
        else:
            print('%-6s : 无有效值' % m)
    print(' '.join(summary_line))

    # 保存 csv
    csv_path = args.csv or os.path.join(img_dir, 'metrics_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name'] + metrics)
        for r in rows:
            writer.writerow([r['name']] + ['%.6f' % r[m] for m in metrics])
        writer.writerow(['mean'] + ['%.6f' % summary[m] for m in metrics])
        writer.writerow(['std'] + ['%.6f' % np.std([r[m] for r in rows
                                                    if not np.isnan(r[m])])
                                    for m in metrics])
    print('\n结果已保存至: %s' % csv_path)


if __name__ == '__main__':
    main()
