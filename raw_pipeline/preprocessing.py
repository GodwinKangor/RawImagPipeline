# Author: Godwin Kangor
# Date: 2026-09-26
# File name: preprocessing.py

from .error_handling import log_error

import numpy as np


def linearize(mosaic, black_level, white_level):
    # load_mosaic and extract_metadata have already validated the samples and levels.
    samples = np.asarray(mosaic, dtype=np.float64)
    black, white = float(black_level), float(white_level)
    statistics = {
        "black_fraction": float(np.mean(samples <= black)),
        "saturation_fraction": float(np.mean(samples >= white)),
    }
    return np.clip((samples - black) / (white - black), 0, 1), statistics


# G1 and G2 are the green sites in row-major order.
def get_bayer_layout(pattern):
    layouts = {
        "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
        "GRBG": {"R": (0, 1), "G1": (0, 0), "G2": (1, 1), "B": (1, 0)},
        "BGGR": {"R": (1, 1), "G1": (0, 1), "G2": (1, 0), "B": (0, 0)},
        "GBRG": {"R": (1, 0), "G1": (0, 0), "G2": (1, 1), "B": (0, 1)},
    }
    if not isinstance(pattern, str) or pattern.upper() not in layouts:
        log_error("Pattern must be RGGB, GRBG, BGGR, or GBRG.")
    return layouts[pattern.upper()]


def print_bayer_layouts(selected_pattern=None):
    if selected_pattern is not None:
        get_bayer_layout(selected_pattern)
        selected_pattern = selected_pattern.upper()

    print("\nBayer layouts (top-left 2 x 2 pixels):")
    for pattern in ("RGGB", "GRBG", "BGGR", "GBRG"):
        grid = [["", ""], ["", ""]]
        for name, (row, column) in get_bayer_layout(pattern).items():
            grid[row][column] = "G" if name in ("G1", "G2") else name
        label = " (current candidate)" if pattern == selected_pattern else ""
        print(f"\n{pattern}{label}")
        for row in grid:
            print(" ".join(row))
    print()


# Return sample views and original coordinates; copy views before changing them.
def extract_lattices(mosaic, pattern):
    samples = np.asarray(mosaic)
    height, width = samples.shape
    lattices = {}
    for name, (row, column) in get_bayer_layout(pattern).items():
        lattices[name] = {
            "samples": samples[row::2, column::2],
            "rows": np.arange(row, height, 2),
            "columns": np.arange(column, width, 2),
        }
    return lattices


def create_channel_masks(image_shape, pattern):
    masks = {color: np.zeros(image_shape, dtype=bool) for color in "RGB"}
    for name, (row, column) in get_bayer_layout(pattern).items():
        color = "G" if name in ("G1", "G2") else name
        masks[color][row::2, column::2] = True
    return masks
