"""Evaluate IVIF results with six metrics used by the FSATFusion paper.

Metrics (all higher is better): MI, NCIE, Qabf, Qp, Qy and VIF.

The implementation follows the common image-fusion metric definitions:
  - MI: sum of mutual information with the two source images
  - NCIE: nonlinear correlation information entropy
  - Qabf: Xydeas edge-information preservation metric
  - Qp: Zhao phase-congruency metric (4 scales, 6 orientations)
  - Qy: Yang local-SSIM fusion metric
  - VIF: sum of two four-scale pairwise VIF values

The script evaluates all methods per image in one worker so that the expensive
source-image phase-congruency maps are reused across methods.
"""

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import signal
from tqdm import tqdm


METRICS = ("MI", "NCIE", "Qabf", "Qp", "Qy", "VIF")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate six IVIF metrics.")
    parser.add_argument(
        "--ir_dir", type=Path,
        required=True,
    )
    parser.add_argument(
        "--vi_dir", type=Path,
        required=True,
    )
    parser.add_argument(
        "--methods", nargs="+", required=True,
        help="Method specifications in tag=folder form."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected_count", type=int, default=361)
    parser.add_argument(
        "--skip_qp", action="store_true",
        help="Skip the expensive Qp metric (diagnostic use only)."
    )
    return parser.parse_args()


def parse_methods(specs):
    methods = []
    tags = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid method specification: {spec}")
        tag, folder = spec.split("=", 1)
        tag = tag.strip()
        if not tag or tag in tags:
            raise ValueError(f"Empty or duplicate method tag: {tag}")
        tags.add(tag)
        methods.append((tag, str(Path(folder.strip()))))
    return methods


def index_images_by_stem(folder):
    indexed = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.lower()
        if key in indexed:
            raise RuntimeError(
                f"Duplicate image stem in {folder}: {indexed[key].name} and {path.name}"
            )
        indexed[key] = path
    return indexed


def collect_tasks(ir_dir, vi_dir, methods, limit, expected_count):
    if not ir_dir.is_dir() or not vi_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {ir_dir} or {vi_dir}")
    infrared = index_images_by_stem(ir_dir)
    visible = index_images_by_stem(vi_dir)
    ir_names = set(infrared)
    vi_names = set(visible)
    if ir_names != vi_names:
        raise RuntimeError(
            f"IR/VI mismatch: only_ir={len(ir_names-vi_names)}, "
            f"only_vi={len(vi_names-ir_names)}"
        )
    stems = sorted(ir_names)
    if limit > 0:
        stems = stems[:limit]
    elif expected_count > 0 and len(stems) != expected_count:
        raise RuntimeError(f"Expected {expected_count} pairs, found {len(stems)}")
    if not stems:
        raise RuntimeError("No paired source images found")

    output_names = {stem: f"{infrared[stem].stem}.png" for stem in stems}

    for tag, folder_string in methods:
        folder = Path(folder_string)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing result folder for {tag}: {folder}")
        missing = [
            output_names[stem]
            for stem in stems
            if not (folder / output_names[stem]).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Method {tag} is missing {len(missing)} files; first={missing[0]}"
            )
    return [
        (
            output_names[stem],
            str(infrared[stem]),
            str(visible[stem]),
            methods,
        )
        for stem in stems
    ]


def load_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def normalize_255(image):
    low = float(np.min(image))
    high = float(np.max(image))
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image - low) / (high - low)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)


def entropy_from_probability(probability):
    values = probability[probability > 0]
    return float(-np.sum(values * np.log2(values)))


def mutual_information_uint8(first, second):
    first_flat = first.reshape(-1).astype(np.int64)
    second_flat = second.reshape(-1).astype(np.int64)
    joint = np.bincount(
        first_flat * 256 + second_flat, minlength=256 * 256
    ).reshape(256, 256).astype(np.float64)
    joint /= joint.sum()
    first_probability = joint.sum(axis=1)
    second_probability = joint.sum(axis=0)
    h_first = entropy_from_probability(first_probability)
    h_second = entropy_from_probability(second_probability)
    h_joint = entropy_from_probability(joint.reshape(-1))
    return h_first + h_second - h_joint


def metric_mi_and_ncie(ir, vi, fused):
    ir_n = normalize_255(ir)
    vi_n = normalize_255(vi)
    fused_n = normalize_255(fused)
    mi_ir_vi = mutual_information_uint8(ir_n, vi_n)
    mi_ir_f = mutual_information_uint8(ir_n, fused_n)
    mi_vi_f = mutual_information_uint8(vi_n, fused_n)
    mi = mi_ir_f + mi_vi_f

    r = np.array(
        [
            [1.0, mi_ir_vi / 8.0, mi_ir_f / 8.0],
            [mi_ir_vi / 8.0, 1.0, mi_vi_f / 8.0],
            [mi_ir_f / 8.0, mi_vi_f / 8.0, 1.0],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.clip(np.linalg.eigvalsh(r), 0.0, None)
    probabilities = eigenvalues / 3.0
    nonzero = probabilities > 0
    ncie = 1.0 + float(
        np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])) / 8.0
    )
    return mi, ncie


def conv_same(image, kernel):
    return signal.convolve2d(
        image, kernel, mode="same", boundary="fill", fillvalue=0
    )


def conv_valid(image, kernel):
    return signal.convolve2d(image, kernel, mode="valid")


def sobel_maps(image):
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sx = conv_same(image, h3)
    sy = conv_same(image, h1)
    magnitude = np.sqrt(sx * sx + sy * sy)
    angle = np.empty_like(sx)
    zero = sx == 0
    angle[zero] = np.pi / 2.0
    angle[~zero] = np.arctan(sy[~zero] / sx[~zero])
    return magnitude, angle


def qabf_quality(source_magnitude, source_angle, fused_magnitude, fused_angle):
    relative_magnitude = np.empty_like(source_magnitude)
    greater = source_magnitude > fused_magnitude
    equal = source_magnitude == fused_magnitude
    lower = ~(greater | equal)
    relative_magnitude[greater] = (
        fused_magnitude[greater] / source_magnitude[greater]
    )
    relative_magnitude[equal] = fused_magnitude[equal]
    relative_magnitude[lower] = (
        source_magnitude[lower] / fused_magnitude[lower]
    )
    relative_angle = 1.0 - np.abs(source_angle - fused_angle) / (np.pi / 2.0)
    qg = 0.9994 / (1.0 + np.exp(-15.0 * (relative_magnitude - 0.5)))
    qa = 0.9879 / (1.0 + np.exp(-22.0 * (relative_angle - 0.8)))
    return qg * qa


def metric_qabf(ir, vi, fused):
    g_ir, a_ir = sobel_maps(ir)
    g_vi, a_vi = sobel_maps(vi)
    g_f, a_f = sobel_maps(fused)
    q_ir = qabf_quality(g_ir, a_ir, g_f, a_f)
    q_vi = qabf_quality(g_vi, a_vi, g_f, a_f)
    denominator = np.sum(g_ir + g_vi)
    if denominator == 0:
        return 0.0
    return float(np.sum(q_ir * g_ir + q_vi * g_vi) / denominator)


def gaussian_kernel(size, sigma):
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    one_d = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    one_d /= one_d.sum()
    kernel = np.outer(one_d, one_d)
    return kernel / kernel.sum()


def ssim_yang(first, second):
    window = gaussian_kernel(7, 1.5)
    mu_first = conv_valid(first, window)
    mu_second = conv_valid(second, window)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_both = mu_first * mu_second
    variance_first = conv_valid(first * first, window) - mu_first_sq
    variance_second = conv_valid(second * second, window) - mu_second_sq
    covariance = conv_valid(first * second, window) - mu_both
    c1 = 2e-16
    c2 = 2e-16
    numerator1 = 2.0 * mu_both + c1
    numerator2 = 2.0 * covariance + c2
    denominator1 = mu_first_sq + mu_second_sq + c1
    denominator2 = variance_first + variance_second + c2
    ssim_map = np.ones_like(mu_first)
    both_positive = denominator1 * denominator2 > 0
    ssim_map[both_positive] = (
        numerator1[both_positive] * numerator2[both_positive]
    ) / (denominator1[both_positive] * denominator2[both_positive])
    first_only = (denominator1 != 0) & (denominator2 == 0)
    ssim_map[first_only] = numerator1[first_only] / denominator1[first_only]
    return ssim_map, variance_first, variance_second


def stable_ssim_yang(first, second):
    window = gaussian_kernel(7, 1.5)
    mu_first = conv_valid(first, window)
    mu_second = conv_valid(second, window)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    variance_first = np.maximum(
        conv_valid(first * first, window) - mu_first_sq, 0.0
    )
    variance_second = np.maximum(
        conv_valid(second * second, window) - mu_second_sq, 0.0
    )
    covariance = conv_valid(first * second, window) - mu_first * mu_second
    covariance_bound = np.sqrt(variance_first * variance_second)
    covariance = np.clip(covariance, -covariance_bound, covariance_bound)
    c1 = 2e-16
    c2 = 2e-16
    luminance = (2.0 * mu_first * mu_second + c1) / (
        mu_first_sq + mu_second_sq + c1
    )
    structure = (2.0 * covariance + c2) / (
        variance_first + variance_second + c2
    )
    return (
        np.clip(luminance * structure, -1.0, 1.0),
        variance_first,
        variance_second,
    )


def metric_qy(ir, vi, fused):
    source_similarity, variance_ir, variance_vi = stable_ssim_yang(ir, vi)
    ir_similarity, _, _ = stable_ssim_yang(ir, fused)
    vi_similarity, _, _ = stable_ssim_yang(vi, fused)
    similar_sources = source_similarity >= 0.75
    variance_sum = variance_ir + variance_vi
    weight = np.full(variance_sum.shape, 0.5, dtype=np.float64)
    np.divide(
        variance_ir,
        variance_sum,
        out=weight,
        where=variance_sum > np.finfo(np.float64).eps,
    )
    similar_quality = weight * ir_similarity + (1.0 - weight) * vi_similarity
    dissimilar_quality = np.maximum(ir_similarity, vi_similarity)
    quality = np.where(similar_sources, similar_quality, dissimilar_quality)
    return float(np.mean(np.clip(quality, -1.0, 1.0)))


def pairwise_vif(reference, distorted, sigma_n_sq=2.0):
    reference = reference.astype(np.float64, copy=True)
    distorted = distorted.astype(np.float64, copy=True)
    numerator_total = 0.0
    denominator_total = 0.0
    epsilon = 1e-10
    for scale in range(1, 5):
        size = 2 ** (4 - scale + 1) + 1
        window = gaussian_kernel(size, size / 5.0)
        if scale > 1:
            reference = conv_valid(reference, window)[::2, ::2]
            distorted = conv_valid(distorted, window)[::2, ::2]
        mu_reference = conv_valid(reference, window)
        mu_distorted = conv_valid(distorted, window)
        variance_reference = np.maximum(
            conv_valid(reference * reference, window) - mu_reference * mu_reference,
            0.0,
        )
        variance_distorted = np.maximum(
            conv_valid(distorted * distorted, window) - mu_distorted * mu_distorted,
            0.0,
        )
        covariance = (
            conv_valid(reference * distorted, window)
            - mu_reference * mu_distorted
        )
        gain = covariance / (variance_reference + epsilon)
        residual_variance = variance_distorted - gain * covariance

        mask = variance_reference < epsilon
        gain[mask] = 0.0
        residual_variance[mask] = variance_distorted[mask]
        variance_reference[mask] = 0.0
        mask = variance_distorted < epsilon
        gain[mask] = 0.0
        residual_variance[mask] = 0.0
        mask = gain < 0
        residual_variance[mask] = variance_distorted[mask]
        gain[mask] = 0.0
        residual_variance = np.maximum(residual_variance, epsilon)

        numerator_total += float(
            np.sum(
                np.log10(
                    1.0
                    + gain * gain * variance_reference
                    / (residual_variance + sigma_n_sq)
                )
            )
        )
        denominator_total += float(
            np.sum(np.log10(1.0 + variance_reference / sigma_n_sq))
        )
    if denominator_total == 0:
        return 1.0
    return numerator_total / denominator_total


def metric_vif(ir, vi, fused):
    return pairwise_vif(ir, fused) + pairwise_vif(vi, fused)


def lowpass_filter(rows, cols, cutoff=0.45, order=15):
    x = (np.arange(cols, dtype=np.float64) - math.floor(cols / 2.0)) / cols
    y = (np.arange(rows, dtype=np.float64) - math.floor(rows / 2.0)) / rows
    radius = np.sqrt(x[np.newaxis, :] ** 2 + y[:, np.newaxis] ** 2)
    return np.fft.fftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * order)))


@lru_cache(maxsize=4)
def phase_filter_bank(rows, cols):
    nscale = 4
    norient = 6
    minimum_wavelength = 3.0
    multiplier = 2.1
    sigma_on_f = 0.55
    dtheta_on_sigma = 1.2
    theta_sigma = np.pi / norient / dtheta_on_sigma

    if cols % 2:
        x_range = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        x_range = np.arange(-cols / 2, cols / 2) / cols
    if rows % 2:
        y_range = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        y_range = np.arange(-rows / 2, rows / 2) / rows
    x, y = np.meshgrid(x_range, y_range)
    radius = np.sqrt(x * x + y * y)
    radius[math.floor(rows / 2), math.floor(cols / 2)] = 1.0
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    sine_theta = np.sin(theta)
    cosine_theta = np.cos(theta)
    lowpass = lowpass_filter(rows, cols)

    log_gabor = []
    for scale in range(nscale):
        wavelength = minimum_wavelength * multiplier ** scale
        center_frequency = 1.0 / wavelength
        radial = np.exp(
            -(np.log(radius / center_frequency) ** 2)
            / (2.0 * np.log(sigma_on_f) ** 2)
        )
        radial *= lowpass
        radial[0, 0] = 0.0
        log_gabor.append(radial)

    all_filters = []
    em_values = []
    noise_bases = []
    for orientation in range(norient):
        angle = orientation * np.pi / norient
        ds = sine_theta * np.cos(angle) - cosine_theta * np.sin(angle)
        dc = cosine_theta * np.cos(angle) + sine_theta * np.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta * dtheta) / (2.0 * theta_sigma * theta_sigma))
        filters = tuple(radial * spread for radial in log_gabor)
        inverse_filters = [
            np.real(np.fft.ifft2(item)) * np.sqrt(rows * cols)
            for item in filters
        ]
        sum_an2 = np.zeros((rows, cols), dtype=np.float64)
        for item in inverse_filters:
            sum_an2 += item * item
        sum_aiaj = np.zeros((rows, cols), dtype=np.float64)
        for first in range(nscale - 1):
            for second in range(first + 1, nscale):
                sum_aiaj += inverse_filters[first] * inverse_filters[second]
        noise_base = 2.0 * float(np.sum(sum_an2)) + 4.0 * float(np.sum(sum_aiaj))
        all_filters.append(filters)
        em_values.append(float(np.sum(filters[0] * filters[0])))
        noise_bases.append(noise_base)
    return tuple(all_filters), tuple(em_values), tuple(noise_bases)


def phase_congruency_features(image):
    image = image.astype(np.float64, copy=False)
    rows, cols = image.shape
    filters_by_orientation, em_values, noise_bases = phase_filter_bank(rows, cols)
    nscale = 4
    norient = 6
    epsilon = 0.0001
    k = 2.0
    cutoff = 0.5
    sharpness = 10.0
    zero = np.zeros((rows, cols), dtype=np.float64)
    total_energy = zero.copy()
    total_amplitude = zero.copy()
    covariance_x2 = zero.copy()
    covariance_y2 = zero.copy()
    covariance_xy = zero.copy()
    image_fft = np.fft.fft2(image)

    for orientation, filters in enumerate(filters_by_orientation):
        responses = [np.fft.ifft2(image_fft * item) for item in filters]
        amplitudes = [np.abs(item) for item in responses]
        sum_amplitude = np.sum(amplitudes, axis=0)
        sum_even = np.sum([np.real(item) for item in responses], axis=0)
        sum_odd = np.sum([np.imag(item) for item in responses], axis=0)
        max_amplitude = np.maximum.reduce(amplitudes)
        vector_energy = np.sqrt(sum_even * sum_even + sum_odd * sum_odd) + epsilon
        mean_even = sum_even / vector_energy
        mean_odd = sum_odd / vector_energy
        energy = zero.copy()
        for response in responses:
            even = np.real(response)
            odd = np.imag(response)
            energy += (
                even * mean_even
                + odd * mean_odd
                - np.abs(even * mean_odd - odd * mean_even)
            )

        median_energy_squared = float(np.median(amplitudes[0] ** 2))
        mean_energy_squared = -median_energy_squared / np.log(0.5)
        noise_power = mean_energy_squared / em_values[orientation]
        estimated_noise_energy2 = noise_power * noise_bases[orientation]
        tau = np.sqrt(max(estimated_noise_energy2, 0.0) / 2.0)
        expected_noise = tau * np.sqrt(np.pi / 2.0)
        noise_sigma = np.sqrt((2.0 - np.pi / 2.0) * tau * tau)
        threshold = (expected_noise + k * noise_sigma) / 1.7
        energy = np.maximum(energy - threshold, 0.0)
        width = sum_amplitude / (max_amplitude + epsilon) / nscale
        weight = 1.0 / (1.0 + np.exp((cutoff - width) * sharpness))
        weighted_energy = weight * energy
        total_amplitude += sum_amplitude
        total_energy += weighted_energy
        with np.errstate(divide="ignore", invalid="ignore"):
            pc = np.divide(
                weighted_energy,
                sum_amplitude,
                out=np.zeros_like(weighted_energy),
                where=sum_amplitude != 0,
            )
        angle = orientation * np.pi / norient
        cov_x = pc * np.cos(angle)
        cov_y = pc * np.sin(angle)
        covariance_x2 += cov_x * cov_x
        covariance_y2 += cov_y * cov_y
        covariance_xy += cov_x * cov_y

    phase_congruency = total_energy / (total_amplitude + epsilon)
    covariance_x2 /= norient / 2.0
    covariance_y2 /= norient / 2.0
    covariance_xy /= norient
    denominator = (
        np.sqrt(covariance_xy * covariance_xy + (covariance_x2 - covariance_y2) ** 2)
        + epsilon
    )
    maximum_moment = (covariance_y2 + covariance_x2 + denominator) / 2.0
    minimum_moment = (covariance_y2 + covariance_x2 - denominator) / 2.0
    return phase_congruency, maximum_moment, minimum_moment


def qp_correlation(first, second, maximum, fused, mask_first, mask_second, mask_maximum):
    window = gaussian_kernel(11, 1.5)
    first = first * mask_first
    second = second * mask_second
    maximum = maximum * mask_maximum
    means = [conv_same(item, window) for item in (first, second, maximum, fused)]
    mean_first, mean_second, mean_maximum, mean_fused = means
    variance_fused = conv_same(fused * fused, window) - mean_fused * mean_fused

    result_maps = []
    for source, mean_source, mask in (
        (first, mean_first, mask_first),
        (second, mean_second, mask_second),
        (maximum, mean_maximum, mask_maximum),
    ):
        variance_source = conv_same(source * source, window) - mean_source * mean_source
        covariance = conv_same(source * fused, window) - mean_source * mean_fused
        result = np.zeros_like(source)
        result[mask] = (covariance[mask] + 0.0001) / (
            np.sqrt(np.abs(variance_source[mask] * variance_fused[mask])) + 0.0001
        )
        result_maps.append(result)
    maximum_result = np.maximum.reduce(result_maps)
    area = int(np.sum(mask_maximum))
    return float(np.sum(maximum_result) / area) if area else 0.0


def metric_qp_from_features(ir_features, vi_features, fused_features):
    pc_ir, maximum_ir, minimum_ir = ir_features
    pc_vi, maximum_vi, minimum_vi = vi_features
    pc_f, maximum_f, minimum_f = fused_features
    choose_ir = pc_ir > pc_vi
    pc_max = np.where(choose_ir, pc_ir, pc_vi)
    maximum_max = np.where(choose_ir, maximum_ir, maximum_vi)
    minimum_max = np.where(choose_ir, minimum_ir, minimum_vi)
    mask_ir = pc_ir > 0.1
    mask_vi = pc_vi > 0.1
    mask_max = pc_max > 0.1
    pc_result = qp_correlation(
        pc_ir, pc_vi, pc_max, pc_f, mask_ir, mask_vi, mask_max
    )
    maximum_result = qp_correlation(
        maximum_ir, maximum_vi, maximum_max, maximum_f,
        mask_ir, mask_vi, mask_max,
    )
    minimum_result = qp_correlation(
        minimum_ir, minimum_vi, minimum_max, minimum_f,
        mask_ir, mask_vi, mask_max,
    )
    return pc_result * maximum_result * minimum_result


def evaluate_task(task_and_skip):
    task, skip_qp = task_and_skip
    name, ir_path, vi_path, methods = task
    ir = load_gray(ir_path)
    vi = load_gray(vi_path)
    if ir.shape != vi.shape:
        raise ValueError(f"Source size mismatch for {name}: {ir.shape} vs {vi.shape}")
    if not skip_qp:
        ir_phase = phase_congruency_features(ir)
        vi_phase = phase_congruency_features(vi)
    rows = []
    for tag, folder in methods:
        fused = load_gray(Path(folder) / name)
        if fused.shape != ir.shape:
            raise ValueError(
                f"Fused size mismatch for {tag}/{name}: {fused.shape} vs {ir.shape}"
            )
        mi, ncie = metric_mi_and_ncie(ir, vi, fused)
        qabf = metric_qabf(ir, vi, fused)
        qy = metric_qy(ir, vi, fused)
        vif = metric_vif(ir, vi, fused)
        if skip_qp:
            qp = float("nan")
        else:
            qp = metric_qp_from_features(
                ir_phase, vi_phase, phase_congruency_features(fused)
            )
        values = (mi, ncie, qabf, qp, qy, vif)
        if not all(np.isfinite(value) for value in values if not np.isnan(value)):
            raise FloatingPointError(f"Non-finite metric for {tag}/{name}: {values}")
        rows.append(
            {
                "image": name,
                "method": tag,
                "MI": mi,
                "NCIE": ncie,
                "Qabf": qabf,
                "Qp": qp,
                "Qy": qy,
                "VIF": vif,
            }
        )
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, method_order):
    summary = []
    for method in method_order:
        selected = [row for row in rows if row["method"] == method]
        output = {"method": method, "images": len(selected)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            output[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            output[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        summary.append(output)
    return summary


def main():
    args = parse_args()
    methods = parse_methods(args.methods)
    tasks = collect_tasks(
        args.ir_dir, args.vi_dir, methods, args.limit, args.expected_count
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    print("=" * 72)
    print("IVIF six-metric evaluation")
    print(f"Images     = {len(tasks)}")
    print(f"Methods    = {', '.join(tag for tag, _ in methods)}")
    print(f"Metrics    = {', '.join(METRICS)}")
    print(f"Workers    = {workers}")
    print(f"Qp enabled = {not args.skip_qp}")
    print(f"Output     = {args.output_dir}")
    print("=" * 72)

    task_inputs = [(task, args.skip_qp) for task in tasks]
    all_rows = []
    if workers == 1:
        iterator = (evaluate_task(item) for item in task_inputs)
        for rows in tqdm(iterator, total=len(task_inputs), desc="metrics"):
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(evaluate_task, task_inputs, chunksize=1)
            for rows in tqdm(iterator, total=len(task_inputs), desc="metrics"):
                all_rows.extend(rows)

    method_order = [tag for tag, _ in methods]
    method_index = {tag: index for index, tag in enumerate(method_order)}
    all_rows.sort(key=lambda row: (row["image"], method_index[row["method"]]))
    per_image_path = args.output_dir / "per_image_metrics.csv"
    write_csv(per_image_path, all_rows, ["image", "method", *METRICS])

    full_summary = summarize(all_rows, method_order)
    summary_fields = ["method", "images"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std")
    ]
    full_path = args.output_dir / "summary_full.csv"
    write_csv(full_path, full_summary, summary_fields)

    print("\nFULL-DATASET MEANS")
    for row in full_summary:
        values = " ".join(f"{metric}={row[f'{metric}_mean']:.6f}" for metric in METRICS)
        print(f"{row['method']}: {values}")
    print(f"\nPer-image CSV = {per_image_path}")
    print(f"Full summary  = {full_path}")
    print("METRIC EVALUATION PASS")


if __name__ == "__main__":
    main()
