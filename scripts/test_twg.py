"""Sanity checks for the 5-level U-Net + TWG (FBDG) changes.

Run from anywhere:
    /home/zd/anaconda3/envs/cold/bin/python scripts/test_twg.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from types import SimpleNamespace

from models.diffusion import Model, WaveletGatedSkip


def make_config(ch=32, ch_mult=(1, 1, 2, 2, 4, 4), twg_levels=()):
    model = SimpleNamespace(
        type='cold_diffusion', in_channels=2, out_ch=1, ch=ch,
        ch_mult=list(ch_mult), num_res_blocks=2, attn_resolutions=[22],
        dropout=0.0, var_type='fixedsmall', ema_rate=0.999, ema=True,
        resamp_with_conv=True, twg_levels=list(twg_levels))
    data = SimpleNamespace(image_size=352)
    diffusion = SimpleNamespace(num_diffusion_timesteps=10)
    return SimpleNamespace(model=model, data=data, diffusion=diffusion)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    device = 'cpu'

    # ---- 1. Haar DWT/IDWT perfect reconstruction -------------------------
    print("[1] Haar DWT/IDWT reconstruction")
    for C in (1, 3, 32):
        m = WaveletGatedSkip(C, temb_channels=128)
        x = torch.randn(2, C, 64, 64)
        ll, lh, hl, hh = m.dwt(x)
        y = m.idwt(ll, lh, hl, hh)
        err = (y - x).abs().max().item()
        assert y.shape == x.shape
        assert err < 1e-5, f"reconstruction error too large: {err}"
        print(f"    C={C}: shape OK, max_abs_err={err:.2e}")

    # ---- 2. TWG init is close to the raw skip ----------------------------
    print("[2] TWG init state")
    m = WaveletGatedSkip(32, temb_channels=128)
    x = torch.randn(2, 32, 176, 176)
    temb = torch.randn(2, 128)
    with torch.no_grad():
        y = m(x, temb)
    assert y.shape == x.shape and torch.isfinite(y).all()
    rel = ((y - x).abs().mean() / x.abs().mean()).item()
    print(f"    shape OK, finite, mean|y-x|/mean|x| at init = {rel:.4f}")

    # ---- 3. Full model forward: baseline config --------------------------
    print("[3] Baseline config (6-level, no TWG)")
    cfg = make_config()
    net = Model(cfg).to(device)
    x = torch.randn(1, 2, 352, 352)
    t = torch.tensor([5])
    with torch.no_grad():
        out = net(x, t)
    assert out.shape == (1, 1, 352, 352) and torch.isfinite(out).all()
    print(f"    forward OK, params = {count_params(net)/1e6:.2f} M")
    n_baseline = count_params(net)
    del net

    # ---- 4. Full model forward: new config --------------------------------
    print("[4] New config (5-level, TWG@[352,176,88])")
    cfg = make_config(ch_mult=(1, 2, 3, 4, 5), twg_levels=(0, 1, 2))
    net = Model(cfg).to(device)
    with torch.no_grad():
        out = net(x, t)
    assert out.shape == (1, 1, 352, 352) and torch.isfinite(out).all()
    print(f"    forward OK, params = {count_params(net)/1e6:.2f} M "
          f"({count_params(net)/n_baseline:.2f}x baseline)")

    # additive time conditioning is live at init: output must depend on t
    with torch.no_grad():
        out_t1 = net(x, torch.tensor([1]))
        out_t10 = net(x, torch.tensor([10]))
    diff = (out_t1 - out_t10).abs().mean().item()
    assert diff > 1e-6, "outputs identical across t: timestep conditioning dead?"
    print(f"    mean|out(t=1)-out(t=10)| = {diff:.4f} (nonzero: good)")

    # gradient must reach the zero-init TWG gate temb from a fresh init
    net2 = Model(make_config(ch_mult=(1, 2, 3, 4, 5), twg_levels=(0, 1, 2)))
    out = net2(x, t)
    out.abs().mean().backward()
    g_gate = net2.twg[0].gate_temb.weight.grad
    assert g_gate is not None and g_gate.abs().sum() > 0, \
        "no gradient to TWG gate temb: timestep path untrainable"
    print("    gradient reaches zero-init TWG gate temb (good)")

    # ---- 5. Switches: 5 levels without TWG --------------------------------
    print("[5] Config switches")
    cfg = make_config(ch_mult=(1, 2, 3, 4, 5), twg_levels=())
    net = Model(cfg)
    with torch.no_grad():
        out = net(x, t)
    assert out.shape == (1, 1, 352, 352) and torch.isfinite(out).all()
    print(f"    5-level + no TWG OK, params = {count_params(net)/1e6:.2f} M")

    # ---- 6. Quick timing on GPU if available ------------------------------
    if torch.cuda.is_available():
        print("[6] GPU timing (single 352x352 forward, B=1)")
        cfg = make_config(ch_mult=(1, 2, 3, 4, 5), twg_levels=(0, 1, 2))
        net = Model(cfg).cuda().eval()
        x = x.cuda()
        t = t.cuda()
        with torch.no_grad():
            for _ in range(3):
                net(x, t)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                net(x, t)
            torch.cuda.synchronize()
            print(f"    {(time.time()-t0)/10*1000:.1f} ms/forward, "
                  f"peak mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print("\nAll checks passed.")


if __name__ == '__main__':
    main()
