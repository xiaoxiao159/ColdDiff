#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速批量计算 PSNR / SSIM (只算这两项, 多进程 + CSV 缓存).

用法:
    python eval_psnr_ssim.py            # 计算所有目录并缓存
    python eval_psnr_ssim.py --fast    # 用快速SSIM(均匀窗口), 略有差异
"""
import os
import sys
import glob
import csv
from multiprocessing import Pool

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as sk_ssim

# (名称, 目录, step前缀)
RUNS = [
    ('stage2_linear_500000', '/home/zd/cold/LoDoPaB_10_stage2/image_samples/LoDoPaB/images_fid', '500000'),
    ('stage2_linear_800000', '/home/zd/cold/LoDoPaB_10_stage2/image_samples/LoDoPaB/images_fid', '800000'),
    ('cosine_500000', '/home/zd/cold/LoDoPaB_10_stage2_consine/image_samples/LoDoPaB/images_fid', '500000'),
    ('lodopab10_300000', '/home/zd/cold/LoDoPaB_10/image_samples/LoDoPaB/images_fid', '300000'),
]

FAST = '--fast' in sys.argv


def load(path):
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(np.float64)
    if img.max() > 1.5:
        img /= 255.0
    return img * 255.0


def one(args):
    gt_path, pt_path = args
    gt = load(gt_path)
    pt = load(pt_path)
    mse = np.mean((gt - pt) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    if FAST:
        ssim = sk_ssim(gt, pt, data_range=255.0)
    else:
        ssim = sk_ssim(gt, pt, data_range=255.0, gaussian_weights=True,
                       sigma=1.5, use_sample_covariance=False)
    return psnr, ssim


def main():
    for name, d, pref in RUNS:
        cache = os.path.join(d, f'metrics_{name}.csv')
        stems = {f[:-4] for f in os.listdir(d)
                 if f.startswith(pref + '_') and f.endswith('.png')}
        pairs = sorted(k[:-3] for k in stems
                       if k.endswith('_pt') and k[:-3] + '_gt' in stems)
        if os.path.exists(cache):
            rows = list(csv.DictReader(open(cache)))
            if len(rows) == len(pairs):
                p = np.array([float(r['psnr']) for r in rows])
                s = np.array([float(r['ssim']) for r in rows])
                print(f'{name}: n={len(p)}  (缓存)  '
                      f'PSNR {p.mean():.4f}+-{p.std():.4f}  SSIM {s.mean():.4f}+-{s.std():.4f}')
                continue
        tasks = [(os.path.join(d, k + '_gt.png'), os.path.join(d, k + '_pt.png'))
                 for k in pairs]
        with Pool() as pool:
            vals = pool.map(one, tasks, chunksize=32)
        with open(cache, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['key', 'psnr', 'ssim'])
            for k, (a, b) in zip(pairs, vals):
                w.writerow([k, f'{a:.6f}', f'{b:.6f}'])
        p = np.array([v[0] for v in vals])
        s = np.array([v[1] for v in vals])
        print(f'{name}: n={len(p)}  '
              f'PSNR {p.mean():.4f}+-{p.std():.4f}  SSIM {s.mean():.4f}+-{s.std():.4f}')


if __name__ == '__main__':
    main()
