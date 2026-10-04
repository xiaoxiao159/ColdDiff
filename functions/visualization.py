"""
Visualization and debug save utilities for Cold Diffusion.

Provides functions to save:
  1. Forward degradation trajectory (x_0 through x_T)
  2. Inference loop intermediates (x_t states, stage 1 & stage 2 predictions)
  3. UNet feature maps at each encoder/decoder level

Triggered by --visualize flag; only processes a few selected test samples.
"""

import os
import numpy as np
import torch
from PIL import Image


def tensor_to_image(tensor):
    """
    Convert a [C, H, W] torch tensor in [-1, 1] to a numpy uint8 array.

    For C == 1: returns [H, W] uint8
    For C >  1: returns [H, W, C] uint8 (for multi-channel feature visualization)
    """
    img = tensor.detach().cpu()
    img = (img + 1.0) / 2.0           # [-1, 1] -> [0, 1]
    img = torch.clamp(img, 0.0, 1.0)
    if img.shape[0] == 1:
        img = img.squeeze(0).numpy()      # [H, W]
    else:
        img = img.permute(1, 2, 0).numpy()  # [H, W, C]
    return (img * 255).astype(np.uint8)


def save_image(tensor, filepath):
    """Save a single [C, H, W] tensor as PNG. Creates parent directories."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    arr = tensor_to_image(tensor)
    if arr.ndim == 2:
        mode = 'L'
    elif arr.shape[-1] == 3:
        mode = 'RGB'
    else:
        mode = 'L'
        arr = arr[..., 0] if arr.ndim == 3 else arr
    Image.fromarray(arr, mode=mode).save(filepath)


def save_forward_degradation_trajectory(x_gt, x_img, alpha_bar, T, save_dir, prefix):
    """
    Compute and save the forward degradation trajectory.

    x_t = alpha_bar[t] * x_gt + (1 - alpha_bar[t]) * x_img

    Saves t=0 (NDCT) through t=T (LDCT) as PNG files.

    Args:
        x_gt:      [1, 1, H, W] NDCT ground truth
        x_img:     [1, 1, H, W] LDCT condition image
        alpha_bar: [T+1] CMPD schedule on the same device
        T:         int, total steps
        save_dir:  str, output directory
        prefix:    str, filename prefix
    """
    out_dir = os.path.join(save_dir, "degradation_trajectory")
    os.makedirs(out_dir, exist_ok=True)

    for t_val in range(0, T + 1):
        alpha = alpha_bar[t_val]
        x_t = alpha * x_gt + (1.0 - alpha) * x_img
        label = "NDCT" if t_val == 0 else ("LDCT" if t_val == T else f"t{t_val:02d}")
        filepath = os.path.join(out_dir, f"{prefix}_t{t_val:02d}_{label}.png")
        save_image(x_t[0], filepath)


def save_inference_intermediates(xs, x0_preds, x0_preds_stage2, save_dir, prefix):
    """
    Save the inference loop intermediates from cold_diffusion_improved_sampling().

    Args:
        xs:              list of [B, 1, H, W] tensors — x_t states (length T+1)
        x0_preds:        list of [B, 1, H, W] tensors — stage 1 predictions (length T)
        x0_preds_stage2: list of [B, 1, H, W] tensors — stage 2 predictions (length T)
        save_dir:        str, output directory
        prefix:          str, filename prefix
    """
    T = len(x0_preds)  # T = 10

    # x_t states
    x_t_dir = os.path.join(save_dir, "inference_intermediates", "x_t")
    os.makedirs(x_t_dir, exist_ok=True)
    for i, xt in enumerate(xs):
        # i=0 -> x_T (LDCT), i=T -> x_0 (final state after all improved sampling steps)
        t_label = T - i
        label = "xT_LDCT" if t_label == T else ("x0_denoised" if t_label == 0 else f"xt{t_label:02d}")
        filepath = os.path.join(x_t_dir, f"{prefix}_step{i:02d}_{label}.png")
        save_image(xt[0], filepath)

    # Stage 1 predictions
    s1_dir = os.path.join(save_dir, "inference_intermediates", "x0_pred_stage1")
    os.makedirs(s1_dir, exist_ok=True)
    for i, pred in enumerate(x0_preds):
        t_val = T - i
        filepath = os.path.join(s1_dir, f"{prefix}_step{i:02d}_t{t_val:02d}.png")
        save_image(pred[0], filepath)

    # Stage 2 predictions
    # NOTE: stage 2 refines at t_prev = (stage 1 t) - 1, so its time index is
    # T-1-i, NOT T-i (which is stage 1's t). Using T-i here would mislabel every
    # file by one step (e.g. step00_t10.png would actually hold the t=9 prediction).
    s2_dir = os.path.join(save_dir, "inference_intermediates", "x0_pred_stage2")
    os.makedirs(s2_dir, exist_ok=True)
    for i, pred in enumerate(x0_preds_stage2):
        t_val = T - 1 - i  # stage 2 prediction is made at t_prev = (T - i) - 1
        filepath = os.path.join(s2_dir, f"{prefix}_step{i:02d}_t{t_val:02d}.png")
        save_image(pred[0], filepath)


def save_feature_maps(features_dict, save_dir, prefix):
    """
    Save UNet intermediate feature maps from a forward pass with return_features=True.

    One image per level: the channel-mean projection (all channels averaged
    per pixel, then min-max normalized to [-1, 1]).

    Args:
        features_dict: dict with keys 'down', 'mid', 'up'
            - 'down': list of N tensors [B, C, H, W] (level 0 .. N-1, high->low res)
            - 'mid':  list of 3 tensors [B, C, H, W] (block_1, attn_1, block_2)
            - 'up':   list of N tensors [B, C, H, W] (level N-1 .. 0, low->high res)
            (the first sample [0] is saved; B is the inference batch size)
        save_dir: str, output directory
        prefix:   str, filename prefix

    Filenames use the resolution level as index; down and up share the same
    level numbering, so matching indices sit at the same resolution:
        down: downsampling0.png .. downsampling{N-1}.png  (0 = highest res / clearest)
        mid:  middle0.png middle1.png middle2.png         (block_1, attn_1, block_2)
        up:   upsampling0.png  .. upsampling{N-1}.png      (0 = highest res / clearest)
    """
    base_dir = os.path.join(save_dir, "feature_maps")

    def _save_mean(feat, subdir, label):
        """feat: [C, H, W] tensor -> save its channel-mean projection as one image."""
        out_dir = os.path.join(base_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)

        mean_proj = feat.mean(dim=0, keepdim=True)  # [1, H, W]
        mean_min = mean_proj.min()
        mean_max = mean_proj.max()
        if mean_max > mean_min:
            mean_proj = (mean_proj - mean_min) / (mean_max - mean_min) * 2.0 - 1.0
        else:
            mean_proj = torch.zeros_like(mean_proj)
        filepath = os.path.join(out_dir, f"{prefix}_{label}.png")
        save_image(mean_proj, filepath)

    # Down features: level 0 (high-res) -> level N-1 (low-res)
    # NOTE: features_dict['*'] tensors are [B, C, H, W] (full batch); slice [0].
    for lvl, feat in enumerate(features_dict['down']):
        _save_mean(feat[0], "down", f"downsampling{lvl}")

    # Mid features: block_1, attn_1, block_2
    for i, feat in enumerate(features_dict['mid']):
        _save_mean(feat[0], "mid", f"middle{i}")

    # Up features: collected level N-1 -> 0; label by actual level so that
    # upsampling{i} matches downsampling{i} in resolution.
    num_levels = len(features_dict['up'])
    for lvl, feat in enumerate(features_dict['up']):
        actual_level = num_levels - 1 - lvl
        _save_mean(feat[0], "up", f"upsampling{actual_level}")


def visualize_cold_diffusion(model, x_img, x_gt, alpha_bar, T, vis_base_dir, case_name, ckpt_id):
    """
    Orchestrator: save all three categories of visualization data for one test case.

    Args:
        model:        UNet model (possibly wrapped in DataParallel)
        x_img:        [B, 1, H, W] LDCT condition image
        x_gt:         [B, 1, H, W] NDCT ground truth
        alpha_bar:    [T+1] CMPD schedule on the same device
        T:            int, total sampling steps
        vis_base_dir: str, root visualization output directory
        case_name:    str, identifier for this test case
        ckpt_id:      int/str, checkpoint identifier

    Returns:
        denoised: [B, 1, H, W] final denoised output (same as normal inference)
    """
    from functions.denoising import cold_diffusion_improved_sampling

    case_dir = os.path.join(vis_base_dir, f"{case_name}_{ckpt_id}")
    os.makedirs(case_dir, exist_ok=True)
    prefix = f"{case_name}_{ckpt_id}"

    # ---- 1. Forward degradation trajectory ----
    save_forward_degradation_trajectory(x_gt, x_img, alpha_bar, T, case_dir, prefix)

    # ---- 2. Inference intermediates ----
    denoised, xs, x0_preds, x0_preds_stage2 = cold_diffusion_improved_sampling(
        x_img, alpha_bar, model, T
    )
    save_inference_intermediates(xs, x0_preds, x0_preds_stage2, case_dir, prefix)

    # ---- 3. UNet feature maps ----
    # DataParallel cannot handle dict-of-list return; unwrap to inner model
    if isinstance(model, torch.nn.DataParallel):
        inner_model = model.module
    else:
        inner_model = model

    with torch.no_grad():
        # Use the final denoised output as input to see the UNet's
        # internal representations of the clean reconstruction
        model_input = torch.cat([x_img, denoised], dim=1)  # [B, 2, H, W]
        t_zero = torch.zeros((x_img.shape[0],), device=x_img.device).float()
        _, features = inner_model(model_input, t_zero, return_features=True)

    save_feature_maps(features, case_dir, prefix)

    # ---- 4. Reference images ----
    ref_dir = os.path.join(case_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    save_image(x_img[0], os.path.join(ref_dir, f"{prefix}_LDCT.png"))
    save_image(x_gt[0], os.path.join(ref_dir, f"{prefix}_NDCT.png"))
    save_image(denoised[0], os.path.join(ref_dir, f"{prefix}_denoised.png"))

    return denoised