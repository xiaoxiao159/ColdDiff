"""
Smoke tests for the merged stage-2 residual guidance (from /home/zd/cold)
into the baseline TWG cold-diffusion codebase.

Verifies on a small 5-level-capable model with switches:
  1. loss forward+backward works for all 4 switch combos
     (use_residual_map x detach_stage1)
  2. detach_stage1 does NOT change the forward value (gradient-only switch)
  3. use_residual_map DOES change the forward value (2ch vs 3ch inputs)
  4. no-detach couples gradients: d(loss)/d(x_img) differs with/without
     detach when stage1_weight=0 (isolates the stage-2 feedback path)
  5. sampling runs for both residual settings, deterministic, output shapes ok
  6. Model raises ValueError on in_channels / use_residual_map mismatch
  7. r_t algebraic identity: x_t - D(p,t) == x_{t-1} - D(p,t-1) for p = pred_1

Run:  python scripts/test_stage2_merge.py
"""
import sys
import copy
import torch

sys.path.insert(0, '.')

from models.diffusion import Model
from functions.losses import cold_diffusion_two_stage_loss
from functions.denoising import cold_diffusion_improved_sampling
from runners.schedules import get_cold_alpha_schedule


class Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def make_config(in_channels=3, use_residual_map=True):
    data = Cfg(image_size=64, channels=1)
    model = Cfg(type='cold_diffusion', in_channels=in_channels, out_ch=1,
                ch=32, ch_mult=[1, 2], num_res_blocks=1, attn_resolutions=[],
                dropout=0.0, var_type='fixedsmall', ema=False, ema_rate=0.999,
                resamp_with_conv=True, twg_levels=[0])
    diffusion = Cfg(num_diffusion_timesteps=10)
    cold = Cfg(num_timesteps=10, stage1_weight=0.4, stage2_weight=0.6,
               predict_target='x0', use_residual_map=use_residual_map,
               detach_stage1=False, r_sigma=0.1)
    return Cfg(data=data, model=model, diffusion=diffusion, cold_diffusion=cold)


def make_inputs(seed=0, B=2, requires_grad_img=False):
    torch.manual_seed(seed)
    x_gt = torch.rand(B, 1, 64, 64) * 2 - 1
    x_img = torch.rand(B, 1, 64, 64) * 2 - 1
    if requires_grad_img:
        x_img.requires_grad_(True)
    t = torch.tensor([3, 8], dtype=torch.long)
    return x_img, x_gt, t


def main():
    results = []

    # ---- shared schedule; one model per channel setting (ablations train
    # ---- separate models: 3ch for rm=True, 2ch for rm=False) ----
    torch.manual_seed(42)
    model = Model(make_config(in_channels=3, use_residual_map=True))
    torch.manual_seed(42)
    model2ch = Model(make_config(in_channels=2, use_residual_map=False))
    alpha_bar = get_cold_alpha_schedule('linear', 10)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- 1. all 4 combos forward+backward ----
    x_img, x_gt, t = make_inputs()
    for rm in (False, True):
        for dt in (False, True):
            m = model if rm else model2ch
            for p in m.parameters():
                p.grad = None
            loss = cold_diffusion_two_stage_loss(
                m, x_img, x_gt, t, alpha_bar,
                stage1_weight=0.4, stage2_weight=0.6,
                use_residual_map=rm, detach_stage1=dt, r_sigma=0.1)
            assert torch.isfinite(loss), f"non-finite loss rm={rm} dt={dt}"
            loss.backward()
            n_grad = sum(1 for p in m.parameters() if p.grad is not None
                         and p.grad.abs().sum() > 0)
            assert n_grad > 0, f"no gradients flowed rm={rm} dt={dt}"
            results.append(f"[1] combo rm={rm} dt={dt}: loss={loss.item():.6f}, "
                           f"{n_grad} params with grad  OK")

    # ---- 2. detach does not change forward value ----
    model.zero_grad(set_to_none=True)
    x_img, x_gt, t = make_inputs(seed=1)
    l_dt = cold_diffusion_two_stage_loss(
        model, x_img, x_gt, t, alpha_bar, use_residual_map=True,
        detach_stage1=True).item()
    l_nodt = cold_diffusion_two_stage_loss(
        model, x_img, x_gt, t, alpha_bar, use_residual_map=True,
        detach_stage1=False).item()
    assert abs(l_dt - l_nodt) < 1e-6, f"detach changed forward: {l_dt} vs {l_nodt}"
    results.append(f"[2] detach changes only gradients: {l_dt:.6f} == {l_nodt:.6f}  OK")

    # ---- 3. use_residual_map changes forward value ----
    model.zero_grad(set_to_none=True)
    l_rm = cold_diffusion_two_stage_loss(
        model, x_img, x_gt, t, alpha_bar, use_residual_map=True,
        detach_stage1=True).item()
    l_norm = cold_diffusion_two_stage_loss(
        model2ch, x_img, x_gt, t, alpha_bar, use_residual_map=False,
        detach_stage1=True).item()
    assert abs(l_rm - l_norm) > 1e-8, "rm toggle did not change forward"
    results.append(f"[3] rm toggle changes forward: 3ch={l_rm:.6f} vs 2ch={l_norm:.6f}  OK")

    # ---- 4. no-detach couples gradients (stage-2 path isolated) ----
    model.zero_grad(set_to_none=True)
    x_img, x_gt, t = make_inputs(seed=2, requires_grad_img=True)
    l1 = cold_diffusion_two_stage_loss(
        model, x_img, x_gt, t, alpha_bar, stage1_weight=0.0, stage2_weight=1.0,
        use_residual_map=True, detach_stage1=True)
    g_dt = torch.autograd.grad(l1, x_img, retain_graph=False)[0]
    model.zero_grad(set_to_none=True)
    x_img, x_gt, t = make_inputs(seed=2, requires_grad_img=True)
    l2 = cold_diffusion_two_stage_loss(
        model, x_img, x_gt, t, alpha_bar, stage1_weight=0.0, stage2_weight=1.0,
        use_residual_map=True, detach_stage1=False)
    g_nodt = torch.autograd.grad(l2, x_img)[0]
    diff = (g_dt - g_nodt).abs().max().item()
    assert diff > 1e-8, f"expected coupling difference, got {diff}"
    results.append(f"[4] no-detach couples stage2->stage1: max|dg|={diff:.3e}  OK")

    # ---- 5. sampling: both settings, deterministic, differ across settings ----
    model.eval()
    x_img, _, _ = make_inputs(seed=3)
    out_rm, xs1, p1, p2 = cold_diffusion_improved_sampling(
        x_img, alpha_bar, model, 10, use_residual_map=True, r_sigma=0.1)
    out_rm2, *_ = cold_diffusion_improved_sampling(
        x_img, alpha_bar, model, 10, use_residual_map=True, r_sigma=0.1)
    out_norm, *_ = cold_diffusion_improved_sampling(
        x_img, alpha_bar, model2ch, 10, use_residual_map=False)
    assert out_rm.shape == x_img.shape and out_norm.shape == x_img.shape
    assert torch.equal(out_rm, out_rm2), "sampling not deterministic"
    assert not torch.allclose(out_rm, out_norm), "rm toggle did not change sampling"
    assert len(xs1) == 11 and len(p1) == 10 and len(p2) == 10
    results.append("[5] sampling both settings: shapes/determinism/divergence  OK")

    # ---- 6. Model channel guard ----
    for bad_in, rm in ((2, True), (3, False)):
        try:
            Model(make_config(in_channels=bad_in, use_residual_map=rm))
            raise AssertionError(f"guard did not raise for in={bad_in}, rm={rm}")
        except ValueError as e:
            results.append(f"[6] guard in={bad_in} rm={rm}: raised OK ({str(e)[:52]}...)")

    # ---- 7. r_t identity: x_t - D(p,t) == x_{t-1} - D(p,t-1) ----
    x_img, x_gt, t = make_inputs(seed=4)
    a_t = alpha_bar[t].view(-1, 1, 1, 1)
    a_tm1 = alpha_bar[(t - 1).clamp(min=0)].view(-1, 1, 1, 1)
    x_t = a_t * x_gt + (1 - a_t) * x_img
    p = torch.randn_like(x_gt) * 0.1
    delta = a_tm1 - a_t
    x_tm1 = x_t + delta * (p - x_img)
    lhs = x_t - a_t * p - (1 - a_t) * x_img
    rhs = x_tm1 - a_tm1 * p - (1 - a_tm1) * x_img
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-6, f"r_t identity violated: {err}"
    results.append(f"[7] r_t algebraic identity holds (max err {err:.1e})  OK")

    print(f"\nmodel params: {n_params/1e6:.2f} M")
    print("\n".join(results))
    print("\nALL TESTS PASSED")


if __name__ == '__main__':
    main()
