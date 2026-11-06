#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py 的加速版: 结果与逐对计算一致, 仅并行化.
- PSNR/SSIM/RMSE/VIF/CNR: 多进程 (CPU)
- LPIPS: GPU 批量前向 (同一模型, 仅浮点精度差异)

用法:
    python fast_metrics.py --prefix 600000 --csv out.csv
"""
import argparse
import csv
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch
from skimage.io import imread

import metrics


def _cpu_metrics(args):
    """worker: 计算 (PSNR, SSIM, RMSE, VIF, CNR), LPIPS 由主进程 GPU 批量算"""
    key, pt_path, gt_path = args
    pt = metrics.to_gray(imread(pt_path))
    gt = metrics.to_gray(imread(gt_path))
    if pt.shape != gt.shape:
        return key, None
    return key, (metrics.psnr(gt, pt),
                 metrics.ssim(gt, pt),
                 metrics.rmse(gt, pt),
                 metrics.vifp(gt, pt),
                 metrics.cnr(gt, pt))


def pick_gpu():
    """返回空闲显存最多的 GPU 编号及空闲字节数"""
    n = torch.cuda.device_count()
    best, best_free = 0, -1
    for i in range(n):
        try:
            free, _total = torch.cuda.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free
        except Exception:
            pass
    return best, best_free


def _forward(model, imgs, keep, device):
    """对一批 (gt, pt) 做前向, OOM 时递归减半重试"""
    def to_t(x):
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return t[None].repeat(3, 1, 1)  # (3,H,W)
    if len(imgs) == 0:
        return {}
    in0 = torch.stack([to_t(g) for g, _ in imgs]).to(device)
    in1 = torch.stack([to_t(p) for _, p in imgs]).to(device)
    try:
        vals = model(in0, in1).view(-1).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(imgs) == 1:
            raise
        mid = len(imgs) // 2
        out = _forward(model, imgs[:mid], keep[:mid], device)
        out.update(_forward(model, imgs[mid:], keep[mid:], device))
        return out
    return {k: float(v) for k, v in zip(keep, vals)}


def batch_lpips(pairs, device=None, batch=64):
    """GPU 批量 LPIPS, 返回 {key: value}"""
    import lpips as lpips_lib
    if device is None:
        gi, free = pick_gpu()
        device = 'cuda:%d' % gi
        print('使用 GPU %d (空闲 %.1f GiB)' % (gi, free / 2**30), flush=True)
    model = lpips_lib.LPIPS(net='alex').to(device).eval()
    torch.set_grad_enabled(False)

    out = {}
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        imgs = []
        keep = []
        for key, pt_path, gt_path in chunk:
            pt = metrics.to_gray(imread(pt_path))
            gt = metrics.to_gray(imread(gt_path))
            if pt.shape != gt.shape:
                continue
            keep.append(key)
            imgs.append((gt, pt))  # in0=gt, in1=pt (与 metrics.py 一致)
        if not keep:
            continue
        out.update(_forward(model, imgs, keep, device))
        print('  lpips %d/%d' % (min(i + batch, len(pairs)), len(pairs)),
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    default_dir = '/home/zd/baseline/LoDoPaB_BC/image_samples/LoDoPaB/images_fid'
    ap.add_argument('-d', '--dir', default=default_dir)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--prefix', default=None)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--resume', action='store_true',
                    help='从 CPU 指标 checkpoint 续算 (跳过 CPU 阶段)')
    args = ap.parse_args()

    img_dir = os.path.abspath(args.dir)
    pairs = metrics.collect_pairs(img_dir, prefix=args.prefix)
    if not pairs:
        print('未找到图像对'); sys.exit(1)
    print('共 %d 对, workers=%d' % (len(pairs), args.workers), flush=True)

    mets = ['PSNR', 'SSIM', 'RMSE', 'LPIPS', 'VIF', 'CNR']
    ckpt = (args.csv or os.path.join(img_dir, 'metrics_results.csv')) \
        + '.cpuphase.pkl'

    # ---- CPU 指标并行 (可从 checkpoint 恢复) ----
    cpu = {}
    if args.resume and os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            cpu = pickle.load(f)
        print('从 checkpoint 恢复 CPU 指标: %d 对' % len(cpu), flush=True)
    else:
        with Pool(args.workers) as pool:
            for n, (key, vals) in enumerate(pool.imap_unordered(
                    _cpu_metrics, pairs, chunksize=16), 1):
                cpu[key] = vals
                if n % 200 == 0 or n == len(pairs):
                    print('cpu %d/%d' % (n, len(pairs)), flush=True)
        with open(ckpt, 'wb') as f:
            pickle.dump(cpu, f)
        print('CPU 指标已 checkpoint 至 %s' % ckpt, flush=True)

    # ---- GPU LPIPS 批量 ----
    lp = batch_lpips(pairs)

    rows = []
    for key, _, _ in pairs:
        if cpu.get(key) is None or key not in lp:
            print('[警告] 跳过 %s' % key)
            continue
        p, s, r, v, c = cpu[key]
        rows.append({'name': key, 'PSNR': p, 'SSIM': s, 'RMSE': r,
                     'LPIPS': lp[key], 'VIF': v, 'CNR': c})

    # ---- 汇总 (与 metrics.py 相同格式) ----
    print('\n========== 汇总 (%d 对图像) ==========' % len(rows))
    summary, summary_line = {}, []
    for m in mets:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        summary[m] = float(np.mean(vals)) if vals else np.nan
        if vals:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            print('%-6s : %s +-%s' % (m, '%.4g' % mean, '%.4g' % std))
            summary_line.append('%s=%.4g+-%.4g' % (m.lower(), mean, std))
        else:
            print('%-6s : 无有效值' % m)
    print(' '.join(summary_line))

    csv_path = args.csv or os.path.join(img_dir, 'metrics_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name'] + mets)
        for r in rows:
            w.writerow([r['name']] + ['%.6f' % r[m] for m in mets])
        w.writerow(['mean'] + ['%.6f' % summary[m] for m in mets])
        w.writerow(['std'] + ['%.6f' % np.std([r[m] for r in rows
                                               if not np.isnan(r[m])])
                              for m in mets])
    print('\n结果已保存至: %s' % csv_path)


if __name__ == '__main__':
    main()
