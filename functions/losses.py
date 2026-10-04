import torch
import math
import time
from medpy import metric
import numpy as np
np.bool = np.bool_


def calculate_psnr(img1, img2):
    # img1: img
    # img2: gt
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100.0
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def sr_noise_estimation_loss(model,
                          x_bw: torch.Tensor,
                          x_md: torch.Tensor,
                          x_fw: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(torch.cat([x_bw, x_fw, x], dim=1), t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def sg_noise_estimation_loss(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_img, x], dim=1), t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def cold_diffusion_two_stage_loss(
    model,
    x_img: torch.Tensor,
    x_gt: torch.Tensor,
    t: torch.LongTensor,
    alpha_bar: torch.Tensor,
    stage1_weight=0.4,
    stage2_weight=0.6,
    r_sigma=0.1,
):
    """
    Cold Diffusion two-stage training loss with residual guidance (r_t).

    Stage 1: model predicts x_0 from x_t (degraded image at timestep t)
        x_t = alpha_bar[t] * x_gt + (1 - alpha_bar[t]) * x_img
        input: concat(x_img, x_t, zeros)  -> 3 channels (r channel zero-filled,
               conv_in is shared with stage 2)
        target: x_gt

    Improved Sampling: correct x_t towards t-1 degradation level
        x_hat_0 = model(concat(x_img, x_t, 0), t)
        delta_alpha = alpha_bar[t-1] - alpha_bar[t]
        x_hat_{t-1} = x_t + delta_alpha * (x_hat_0 - x_img)
        (NO detach: stage 2 loss backprops into stage 1 through both
         x_hat_{t-1} and r_t -> joint two-level optimization)

    Residual guidance map:
        r_t = tanh((x_{t-1} - D(x_hat_0, t-1)) / sigma_t)
            = tanh((x_t - D(x_hat_0, t)) / sigma_t)      # algebraically equal
            = tanh(-alpha_bar[t] * (x_hat_0 - x_gt) / sigma_t)   # on clean x_t
        i.e. a normalized per-pixel map of stage 1's prediction error.

    Stage 2: model predicts x_0 from x_hat_{t-1} + r_t
        input: concat(x_img, x_hat_{t-1}, r_t)  -> 3 channels
        target: x_gt

    Total loss: stage1_weight * MSE(stage1_pred, x_gt)
               + stage2_weight * MSE(stage2_pred, x_gt)

    Args:
        model: UNet denoising network
        x_img: [B, 1, H, W] LDCT condition image
        x_gt:  [B, 1, H, W] NDCT target image
        t:     [B] sampled timesteps in [1, T]
        alpha_bar: [T+1] CMPD cumulative alpha schedule
        stage1_weight: float, weight for stage 1 loss (default 0.4)
        stage2_weight: float, weight for stage 2 loss (default 0.6)
        r_sigma: float, tanh normalization scale for r_t (default 0.1)

    Returns:
        loss: scalar tensor
    """
    B, C, H, W = x_img.shape

    # ---- Forward degradation ----
    # alpha_bar[t]: [B, 1, 1, 1]
    alpha_t = alpha_bar[t].view(B, 1, 1, 1).to(x_img.device)

    # x_t = alpha_t * NDCT + (1 - alpha_t) * LDCT
    x_t = alpha_t * x_gt + (1.0 - alpha_t) * x_img

    # ---- Stage 1: Predict clean image from x_t ----
    # Third channel is zero-filled: stage 1 has no residual map yet.
    zero_ch = torch.zeros_like(x_img)
    model_input_1 = torch.cat([x_img, x_t, zero_ch], dim=1)  # [B, 3, H, W]
    x0_pred_1 = model(model_input_1, t.float())

    loss_1 = (x0_pred_1 - x_gt).pow(2).mean(dim=(1, 2, 3)).mean()

    # ---- Improved sampling: degrade back to t-1 level ----
    # Get alpha at t-1 (clamp to avoid negative index). At t=0 the clamp makes
    # delta_alpha=0 and x_hat_tm1 == x_t; r_t = tanh((x_0 - x_hat_0)/sigma)
    # is still well defined, no special handling needed.
    t_minus_1 = (t - 1).clamp(min=0)
    alpha_tm1 = alpha_bar[t_minus_1].view(B, 1, 1, 1).to(x_img.device)

    # delta_alpha = alpha_{t-1} - alpha_t  (positive since alpha decreases)
    delta_alpha = alpha_tm1 - alpha_t

    # x_{t-1} = x_t + delta_alpha * (x_hat_0 - x_T)
    # No detach(): stage 2 loss flows back to stage 1 through x_hat_tm1 (path A)
    x_hat_tm1 = x_t + delta_alpha * (x0_pred_1 - x_img)

    # ---- Residual guidance map r_t ----
    # x_{t-1} - D(x_hat_0, t-1) simplifies to x_t - D(x_hat_0, t); the tanh
    # bounds the map to [-1, 1]. Gradient w.r.t. x_hat_0 carries sech^2 * (-alpha_t / sigma).
    r_t = torch.tanh((x_t - alpha_t * x0_pred_1 - (1.0 - alpha_t) * x_img) / r_sigma)

    # ---- Stage 2: Refined prediction from corrected image + residual map ----
    model_input_2 = torch.cat([x_img, x_hat_tm1, r_t], dim=1)  # [B, 3, H, W]
    x0_pred_2 = model(model_input_2, t_minus_1.float())

    loss_2 = (x0_pred_2 - x_gt).pow(2).mean(dim=(1, 2, 3)).mean()

    # ---- Compound loss ----
    loss = stage1_weight * loss_1 + stage2_weight * loss_2

    return loss


loss_registry = {
    'simple': noise_estimation_loss,
    'sr': sr_noise_estimation_loss,
    'sg': sg_noise_estimation_loss,
    # Note: cold_diffusion_two_stage_loss has a different function signature
    # (model, x_img, x_gt, t, alpha_bar, stage1_weight, stage2_weight)
    # and is called directly in sg_train_cold_diffusion(), NOT via loss_registry.
}