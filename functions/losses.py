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
    use_residual_map=False,
    detach_stage1=True,
    r_sigma=0.1,
):
    """
    Cold Diffusion two-stage training loss with improved sampling.

    Switches (for ablation):
        use_residual_map: stage-2 sees a tanh-normalized per-pixel map of
            stage-1's prediction error as a third input channel
            (r_t = tanh((x_t - D(x_hat_0, t)) / r_sigma)). Stage 1 gets the
            r channel zero-filled so conv_in is shared between stages.
            False -> plain 2-channel inputs (original baseline behavior).
        detach_stage1: cut the gradient from the stage-2 loss back into the
            stage-1 prediction (both through x_hat_{t-1} and r_t). False
            enables the joint two-level optimization. NOTE: this only affects
            gradients, the forward values (and hence the train/inference input
            distribution) are identical either way.
        r_sigma: tanh normalization scale for r_t (only used when
            use_residual_map=True).

    Forward pass (identical for all switch settings):
        x_t = alpha_bar[t] * x_gt + (1 - alpha_bar[t]) * x_img
        Stage 1: x_hat_0 = model([x_img, x_t, (0)], t), target x_gt
        Improved sampling: x_hat_{t-1} = x_t + delta_alpha * (x_hat_0 - x_img)
            with delta_alpha = alpha_bar[t-1] - alpha_bar[t] > 0
        Stage 2: x_hat_hat_0 = model([x_img, x_hat_{t-1}, (r_t)], t-1),
            target x_gt

    Total loss: stage1_weight * MSE(stage1_pred, x_gt)
               + stage2_weight * MSE(stage2_pred, x_gt)

    Args:
        model: UNet denoising network
        x_img: [B, 1, H, W] LDCT condition image
        x_gt:  [B, 1, H, W] NDCT target image
        t:     [B] sampled timesteps in [1, T]
        alpha_bar: [T+1] cumulative degradation schedule
        stage1_weight: float, weight for stage 1 loss
        stage2_weight: float, weight for stage 2 loss
        use_residual_map: bool, feed r_t error map to stage 2 (3ch) or not (2ch)
        detach_stage1: bool, detach stage-1 prediction in the handoff
        r_sigma: float, tanh normalization scale for r_t

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
    if use_residual_map:
        # r channel zero-filled: stage 1 has no residual map yet.
        model_input_1 = torch.cat([x_img, x_t, torch.zeros_like(x_img)], dim=1)
    else:
        model_input_1 = torch.cat([x_img, x_t], dim=1)  # [B, 2, H, W]
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
    # detach_stage1 controls whether stage-2 gradients flow back into stage-1.
    # The handoff covers BOTH x_hat_tm1 and r_t: either both carry gradient
    # (joint optimization) or neither does.
    pred_1 = x0_pred_1.detach() if detach_stage1 else x0_pred_1
    x_hat_tm1 = x_t + delta_alpha * (pred_1 - x_img)

    # ---- Stage 2: Refined prediction from corrected image ----
    if use_residual_map:
        # r_t = tanh((x_t - D(pred_1, t)) / r_sigma); on the clean x_t this
        # simplifies to tanh(-alpha_t * (pred_1 - x_gt) / r_sigma), i.e. the
        # normalized per-pixel stage-1 error map. Bounded to [-1, 1].
        r_t = torch.tanh(
            (x_t - alpha_t * pred_1 - (1.0 - alpha_t) * x_img) / r_sigma
        )
        model_input_2 = torch.cat([x_img, x_hat_tm1, r_t], dim=1)  # [B, 3, H, W]
    else:
        model_input_2 = torch.cat([x_img, x_hat_tm1], dim=1)  # [B, 2, H, W]
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