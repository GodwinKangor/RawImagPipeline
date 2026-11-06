# Author: Godwin Kangor
# Date: 2026-09-26
# File name: color.py

from .error_handling import log_error

import numpy as np
from skimage.color import rgb2gray

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def validate_rgb(image):
    values = np.asarray(image)
    if (values.ndim != 3 or values.shape[2] != 3 or values.size == 0
            or values.dtype.kind not in "uif" or not np.all(np.isfinite(values))):
        log_error("Expected a nonempty, finite numeric H x W x 3 image.")
    return values.astype(np.float64, copy=False)


# Use nine dcraw coefficients in their original scale; normalize rows before inversion.
def build_color_transform(camera_coefficients):
    coefficients = np.asarray(camera_coefficients)
    if (coefficients.shape != (9,) or coefficients.dtype.kind not in "uif"
            or not np.all(np.isfinite(coefficients))):
        log_error("Provide nine finite row-major dcraw coefficients.")
    xyz_to_camera = coefficients.astype(np.float64).reshape(3, 3) / 10000
    srgb_to_camera = xyz_to_camera @ SRGB_TO_XYZ
    row_sums = srgb_to_camera.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(row_sums)) or np.any(np.abs(row_sums) <= 1e-12):
        log_error("Camera matrix rows must have finite, nonzero sums.")
    srgb_to_camera /= row_sums
    condition = np.linalg.cond(srgb_to_camera)
    if not np.isfinite(condition) or condition > 1e12:
        log_error("Camera matrix is singular or too ill-conditioned to invert.")
    return np.linalg.inv(srgb_to_camera)


def correct_color(camera_rgb, camera_to_srgb):
    # camera_to_srgb comes from build_color_transform, which already validated it.
    rgb = validate_rgb(camera_rgb)
    # Image pixels store RGB as row vectors, so transpose the color matrix.
    return rgb @ np.asarray(camera_to_srgb, dtype=np.float64).T


# Scale toward the target mean; clipping highlights may lower the final mean.
def adjust_brightness(linear_srgb, target_mean):
    rgb = np.asarray(linear_srgb, dtype=np.float64)
    if not 0 < target_mean <= 1:
        log_error("Target mean must be in (0, 1].")
    bright = np.maximum(rgb, 0)
    current_mean = float(np.mean(rgb2gray(bright)))
    if current_mean <= 1e-12:
        log_error("Image mean is too close to zero for brightness scaling.")
    scale = float(target_mean / current_mean)
    bright *= scale
    clipped_fraction = float(np.mean(np.any(bright > 1, axis=2)))
    np.clip(bright, 0, 1, out=bright)
    statistics = {
        "target_mean": float(target_mean),
        "original_mean": current_mean,
        "scale": scale,
        "clipped_pixel_fraction": clipped_fraction,
        "negative_pixel_fraction": float(np.mean(np.any(rgb < 0, axis=2))),
        "achieved_mean": float(np.mean(rgb2gray(bright))),
    }
    return bright, statistics


def gamma_encode_srgb(display_linear):
    linear = np.asarray(display_linear, dtype=np.float64)
    if np.any(linear < 0) or np.any(linear > 1):
        log_error("sRGB encoding requires linear values within [0, 1].")
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(encoded, 0, 1)
