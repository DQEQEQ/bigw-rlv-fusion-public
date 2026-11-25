"""BI-GW and RLV for the aligned, single-channel fusion experiments."""

import torch
from torch import nn
from torch.nn import functional as F


_SPATIAL_AND_CHANNEL = (1, 2, 3)


def _local_variance(image):
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    mean = F.avg_pool2d(padded, 3, stride=1)
    mean_square = F.avg_pool2d(padded * padded, 3, stride=1)
    return (mean_square - mean * mean).clamp_min(0.0)


def relative_local_variance_error(infrared, visible, fused):
    """Return one RLV error per image pair, prior to averaging the batch."""
    target = torch.maximum(_local_variance(infrared), _local_variance(visible))
    observed = _local_variance(fused)
    difference = (observed - target).abs().mean(dim=_SPATIAL_AND_CHANNEL)
    return difference / (target.mean(dim=_SPATIAL_AND_CHANNEL) + 1e-4)


def _weighted_score(energy_ir, energy_vi, score_ir, score_vi):
    denominator = energy_ir + energy_vi + 2e-8
    weight_ir = (energy_ir + 1e-8) / denominator
    weight_vi = (energy_vi + 1e-8) / denominator
    return (weight_ir * score_ir + weight_vi * score_vi).mean()


def _ssim_map(first, second, window):
    padding = window.shape[-1] // 2
    mean_first = F.conv2d(first, window, padding=padding)
    mean_second = F.conv2d(second, window, padding=padding)
    first_squared = mean_first.square()
    second_squared = mean_second.square()
    product = mean_first * mean_second
    variance_first = F.conv2d(first * first, window, padding=padding) - first_squared
    variance_second = F.conv2d(second * second, window, padding=padding) - second_squared
    covariance = F.conv2d(first * second, window, padding=padding) - product
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2.0 * product + c1) * (2.0 * covariance + c2)
    denominator = (first_squared + second_squared + c1) * (variance_first + variance_second + c2)
    return numerator / denominator.clamp_min(1e-8)


class SwinFusionSupervision(nn.Module):
    """Use the supplied SwinFusion loss operators with the paper's supervision variants."""

    variants = ("baseline", "bi_eq", "bils", "bigw", "rlv_only", "bigw_rlv")

    def __init__(self, baseline_loss, create_window, variant):
        super().__init__()
        if variant not in self.variants:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {self.variants}.")
        self.baseline = baseline_loss
        self.variant = variant
        self.register_buffer("ssim_window", create_window(11, 1))

    def _score(self, infrared, visible, fused):
        window = self.ssim_window.to(fused)
        score_ir = _ssim_map(infrared, fused, window)
        score_vi = _ssim_map(visible, fused, window)
        if self.variant == "bi_eq":
            return (0.5 * score_ir.mean(dim=_SPATIAL_AND_CHANNEL)
                    + 0.5 * score_vi.mean(dim=_SPATIAL_AND_CHANNEL)).mean()

        gradient_ir = self.baseline.L_SSIM.sobelconv(infrared)
        gradient_vi = self.baseline.L_SSIM.sobelconv(visible)
        if self.variant == "bils":
            energy_ir = F.avg_pool2d(gradient_ir, 11, stride=1, padding=5)
            energy_vi = F.avg_pool2d(gradient_vi, 11, stride=1, padding=5)
            return _weighted_score(energy_ir, energy_vi, score_ir, score_vi)
        return _weighted_score(
            gradient_ir.mean(dim=_SPATIAL_AND_CHANNEL),
            gradient_vi.mean(dim=_SPATIAL_AND_CHANNEL),
            score_ir.mean(dim=_SPATIAL_AND_CHANNEL),
            score_vi.mean(dim=_SPATIAL_AND_CHANNEL),
        )

    def forward(self, image_A, image_B, image_fused):
        if self.variant in ("baseline", "rlv_only"):
            total, gradient, intensity, structural = self.baseline(image_A, image_B, image_fused)
            if self.variant == "baseline":
                return total, gradient, intensity, structural
        else:
            intensity = 20.0 * self.baseline.L_Inten(image_A, image_B, image_fused)
            gradient = 20.0 * self.baseline.L_Grad(image_A, image_B, image_fused)
            structural = 10.0 * (1.0 - self._score(image_A, image_B, image_fused))
        if self.variant in ("rlv_only", "bigw_rlv"):
            structural = structural + relative_local_variance_error(image_A, image_B, image_fused).mean()
        return intensity + gradient + structural, gradient, intensity, structural
