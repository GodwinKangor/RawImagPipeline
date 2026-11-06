# Author: Godwin Kangor
# Date: 2026-09-26
# File name: white_balance.py

from .error_handling import log_error

import numpy as np
import matplotlib.pyplot as plt

from .preprocessing import create_channel_masks


def validate_gains(gains):
    values = np.asarray(gains)
    if values.shape != (3,) or values.dtype.kind not in "uif":
        log_error("Provide three numeric gains in RGB order.")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        log_error("RGB gains must be finite and positive.")
    return values


# Match red and blue maxima to the green maximum.
def white_world_balancing(red, green, blue):
    red_max = np.max(red)
    green_max = np.max(green)
    blue_max = np.max(blue)
    if min(red_max, green_max, blue_max) <= 1e-12:
        log_error("Channel maxima are too close to zero for white balance.")

    red_gain = green_max / red_max
    green_gain = 1.0
    blue_gain = green_max / blue_max
    return validate_gains([red_gain, green_gain, blue_gain])


# Match red and blue means to the green mean.
def gray_world_balancing(red, green, blue):
    red_mean = np.mean(red)
    green_mean = np.mean(green)
    blue_mean = np.mean(blue)
    if min(red_mean, green_mean, blue_mean) <= 1e-12:
        log_error("Channel means are too close to zero for white balance.")

    red_gain = green_mean / red_mean
    green_gain = 1.0
    blue_gain = green_mean / blue_mean
    return validate_gains([red_gain, green_gain, blue_gain])


# Use the camera's RGB white-balance multipliers extracted by dcraw -w.
def camera_balancing(camera_gains):
    return validate_gains(camera_gains)


def automatic_gains(linear_mosaic, pattern, method, camera_gains=None):
    # linearize already returns a finite 2D mosaic clipped to [0, 1].
    samples = np.asarray(linear_mosaic, dtype=np.float64)
    masks = create_channel_masks(samples.shape, pattern)
    if method == "camera":
        return camera_balancing(camera_gains)
    if method not in ("white_world", "gray_world"):
        log_error("Method must be white_world, gray_world, or camera.")

    red = samples[masks["R"]]
    green = samples[masks["G"]]  # Both green Bayer lattices.
    blue = samples[masks["B"]]
    if method == "white_world":
        return white_world_balancing(red, green, blue)
    return gray_world_balancing(red, green, blue)


# Map preview clicks to original-image bounds with exclusive stop indices.
def select_neutral_patch(preview, mosaic_shape):
    preview = np.asarray(preview)
    height, width = mosaic_shape
    preview_height, preview_width = preview.shape[:2]
    figure, axes = plt.subplots()
    try:
        axes.imshow(preview, origin="upper",
                    extent=(0, preview_width, preview_height, 0),
                    cmap="gray" if preview.ndim == 2 else None)
        axes.set_title("Select two corners of a neutral patch; Enter cancels")
        clicks = figure.ginput(2, timeout=0)
    finally:
        plt.close(figure)
    points = np.asarray(clicks, dtype=np.float64)
    if points.shape != (2, 2) or not np.all(np.isfinite(points)):
        log_error("Patch selection cancelled or incomplete.")

    x = np.clip(np.sort(points[:, 0]), 0, preview_width)
    y = np.clip(np.sort(points[:, 1]), 0, preview_height)
    if x[0] == x[1] or y[0] == y[1]:
        log_error("Select a patch with nonzero width and height.")
    return (int(np.floor(y[0] * height / preview_height)),
            int(np.ceil(y[1] * height / preview_height)),
            int(np.floor(x[0] * width / preview_width)),
            int(np.ceil(x[1] * width / preview_width)))


# Estimate neutral-patch gains using the original Bayer phase.
def manual_balance(linear_mosaic, pattern, patch_bounds, min_mean=1e-4,
                 max_saturation_fraction=0.05):
    samples = np.asarray(linear_mosaic, dtype=np.float64)
    # Bounds may come from the config file, so check them against the mosaic.
    row_start, row_stop, column_start, column_stop = map(int, patch_bounds)
    height, width = samples.shape
    if not (0 <= row_start < row_stop <= height
            and 0 <= column_start < column_stop <= width):
        log_error("Patch bounds must describe a nonempty rectangle inside the mosaic.")

    patch_slice = np.s_[row_start:row_stop, column_start:column_stop]
    patch = samples[patch_slice]
    masks = create_channel_masks(samples.shape, pattern)
    means, saturated, counts = {}, {}, {}
    for color in "RGB":
        # Slice masks from the full image to preserve odd-origin Bayer phase.
        channel = patch[masks[color][patch_slice]]
        if channel.size == 0:
            log_error(f"Patch contains no {color} samples; select a larger patch.")
        means[color] = float(np.mean(channel))
        saturated[color] = float(np.mean(channel >= 1))
        counts[color] = int(channel.size)
        if means[color] <= min_mean:
            log_error(f"Patch is too dark in {color}; select a brighter neutral patch.")
        if saturated[color] > max_saturation_fraction:
            log_error(f"Patch is too saturated in {color}; select another patch.")

    gains = validate_gains([means["G"] / means[color] for color in "RGB"])
    return gains, {
        "patch_bounds": (row_start, row_stop, column_start, column_stop),
        "channel_means": means,
        "saturation_fractions": saturated,
        "sample_counts": counts,
    }


# Preserve the input and keep balanced values above one.
def apply_white_balance(linear_mosaic, pattern, gains):
    balanced = np.array(linear_mosaic, dtype=np.float64)  # copy; input stays unchanged
    masks = create_channel_masks(balanced.shape, pattern)
    for color, gain in zip("RGB", validate_gains(gains)):
        balanced[masks[color]] *= gain
    return balanced
