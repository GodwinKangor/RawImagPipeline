# Author: Godwin Kangor
# Date: 2026-09-25
# File name: demosaic.py

from .error_handling import log_error

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .preprocessing import extract_lattices, get_bayer_layout


# Use bilinear interpolation, clamping queries to the sampled boundaries.
# Inputs come from extract_lattices, so rows and columns are already valid.
def interpolate_lattice(samples, rows, columns, image_shape):
    interpolator = RectBivariateSpline(
        rows, columns, samples.astype(np.float64, copy=False), kx=1, ky=1, s=0)
    query_rows = np.clip(np.arange(image_shape[0]), rows[0], rows[-1])
    query_columns = np.clip(np.arange(image_shape[1]), columns[0], columns[-1])
    return interpolator(query_rows, query_columns)


# Average the green estimates, then restore measured sensor samples.
def demosaic(balanced_mosaic, pattern):
    samples = np.asarray(balanced_mosaic)
    # Each lattice needs two samples per axis for bilinear interpolation.
    if samples.ndim != 2 or min(samples.shape) < 4:
        log_error("Expected a Bayer mosaic at least 4 x 4 pixels.")
    lattices = extract_lattices(samples, pattern)
    rgb = np.zeros((*samples.shape, 3), dtype=np.float64)
    # RGB channel index and contribution: green averages two lattice estimates.
    contributions = {"R": (0, 1.0), "G1": (1, 0.5),
                     "G2": (1, 0.5), "B": (2, 1.0)}
    for name, (channel, weight) in contributions.items():
        lattice = lattices[name]
        rgb[:, :, channel] += weight * interpolate_lattice(
            lattice["samples"], lattice["rows"], lattice["columns"], samples.shape)

    # At missing interior green sites the two estimates give the four-neighbor
    # average. Restore all measured samples, including both green lattices.
    for name, (row, column) in get_bayer_layout(pattern).items():
        channel, _ = contributions[name]
        rgb[row::2, column::2, channel] = samples[row::2, column::2]
    return rgb
