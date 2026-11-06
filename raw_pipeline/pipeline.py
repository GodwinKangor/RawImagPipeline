# Author: Godwin Kangor
# Date: 2026-09-26
# File name: pipeline.py

from .error_handling import log_error

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize

from .raw_conversion import prepare_raw
from .preprocessing import get_bayer_layout, linearize, print_bayer_layouts
from .white_balance import automatic_gains, apply_white_balance
from .demosaic import demosaic
from .color import build_color_transform, correct_color, adjust_brightness, gamma_encode_srgb
from .experiments import add_dcraw_comparison, compare_manual_balance, export_bayer_previews


# Resolve relative paths against the configuration file, not the working directory.
def load_config(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        log_error("Configuration must be a JSON object.")
    required = {"raw_path", "work_dir", "pattern", "white_balance", "camera_model",
                "camera_coefficients", "coefficient_source", "target_mean"}
    if not required <= set(config) or set(config) - required - {"camera_path", "manual_patches"}:
        log_error(f"Configuration requires {sorted(required)}; camera_path and "
                  "manual_patches are optional.")
    paths = [name for name in ("raw_path", "work_dir", "camera_path") if name in config]
    for name in (*paths, "camera_model", "coefficient_source"):
        if not isinstance(config[name], str) or not config[name].strip():
            log_error(f"{name} must be a nonempty string.")
    for name in paths:
        # Path / absolute_path returns the absolute path unchanged.
        config[name] = (config_path.parent / Path(config[name]).expanduser()).resolve()
    get_bayer_layout(config["pattern"])
    config["pattern"] = config["pattern"].upper()
    if config["white_balance"] not in ("white_world", "gray_world", "camera"):
        log_error("white_balance must be white_world, gray_world, or camera.")
    build_color_transform(config["camera_coefficients"])
    target = config["target_mean"]
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 < target <= 1:
        log_error("target_mean must be in (0, 1].")
    return config


def downsample(values, step=4):
    shape = (max(1, values.shape[0] // step), max(1, values.shape[1] // step))
    return resize(values, shape + values.shape[2:], preserve_range=True, anti_aliasing=True)


def to_uint8(encoded):
    return np.rint(encoded * 255).astype(np.uint8)


def render_camera_rgb(linear_mosaic, pattern, gains):
    balanced = apply_white_balance(linear_mosaic, pattern, gains)
    return demosaic(balanced, pattern)


def render_image(linear_mosaic, pattern, gains, camera_to_srgb, target_mean):
    camera_rgb = render_camera_rgb(linear_mosaic, pattern, gains)
    linear_srgb = correct_color(camera_rgb, camera_to_srgb)
    del camera_rgb
    bright, measurements = adjust_brightness(linear_srgb, target_mean)
    del linear_srgb
    return gamma_encode_srgb(bright), measurements


def export_stage_outputs(mosaic, linear, metadata, pattern, output_dir,
                         camera_to_srgb, target_mean):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = {"pattern": pattern, "pattern_status": "candidate",
               "metadata": metadata, "preview_scale": 4,
               "note": "Stages 1-4 are diagnostics; stage 5 applies camera color correction."}

    def save_preview(stage, name, values):
        directory = output_dir / stage
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}.png"
        io.imsave(path, to_uint8(np.clip(downsample(values), 0, 1)), check_contrast=False)
        return str(path.resolve())

    # Use saturation as the display divisor ---> this is not black-level correction.
    records["raw_io"] = save_preview("01_raw_io", "sensor_mosaic",
                                      mosaic / metadata["white_level"])
    records["preprocessing"] = save_preview("02_preprocessing", "linear_mosaic", linear)
    records["white_balance"] = {}
    for method in ("white_world", "gray_world", "camera"):
        print(f"Saving module outputs: {method}", flush=True)
        gains = automatic_gains(linear, pattern, method, metadata["white_balance_gains"])
        balanced = apply_white_balance(linear, pattern, gains)
        mosaic_path = save_preview("03_white_balance", method + "_mosaic", balanced)
        rgb = demosaic(balanced, pattern)
        del balanced
        linear_path = save_preview("04_demosaic", method + "_linear_camera_rgb", rgb)
        corrected = correct_color(rgb, camera_to_srgb)
        del rgb
        corrected_path = save_preview("05_color_display", method + "_linear_srgb", corrected)
        bright, stats = adjust_brightness(corrected, target_mean)
        del corrected
        save_preview("05_color_display", method + "_brightness", bright)
        encoded = gamma_encode_srgb(bright)
        del bright
        encoded_path = save_preview("05_color_display", method + "_encoded", encoded)
        exports = export_images(encoded, output_dir.parent / "final" / method)
        del encoded
        # Keep this per-stage JSON independent of the manifest serializer.
        exports = {key: {**value, "path": str(value["path"].resolve())}
                   for key, value in exports.items()}
        records["white_balance"][method] = {
            "gains": gains.tolist(), "mosaic": mosaic_path, "demosaic": linear_path,
            "brightness": stats, "linear_srgb": corrected_path,
            "encoded": encoded_path, "exports": exports}
    records["color_correction"] = {"applied": True, "matrix": camera_to_srgb.tolist()}
    (output_dir / "stages.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def export_images(encoded_rgb, output_stem, jpeg_quality=95):
    encoded = np.asarray(encoded_rgb)
    if encoded.ndim != 3 or encoded.shape[2] != 3 or np.any(encoded < 0) or np.any(encoded > 1):
        log_error("Expected an H x W x 3 encoded image in [0, 1].")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": Path(str(stem) + ".png"), "jpeg": Path(str(stem) + ".jpg")}
    for path in paths.values():
        if path.exists():
            log_error(f"Refusing to overwrite image: {path}")
    rgb8 = to_uint8(encoded)
    io.imsave(paths["png"], rgb8, check_contrast=False)
    io.imsave(paths["jpeg"], rgb8, quality=jpeg_quality, check_contrast=False)
    records = {}
    for name, path in paths.items():
        decoded = io.imread(path)
        if name == "png" and not np.array_equal(decoded, rgb8):
            log_error(f"PNG did not preserve quantized pixels: {path}")
        records[name] = {"path": path, "bytes": path.stat().st_size,
                         "shape": decoded.shape}
    records["jpeg"]["quality"] = jpeg_quality
    return records

def main(config_path):
    config = load_config(config_path)
    print_bayer_layouts(config["pattern"])
    print("Preparing RAW copies, metadata, and Bayer TIFF...", flush=True)
    mosaic, metadata, _, paths = prepare_raw(config["raw_path"], config["work_dir"])
    if metadata["camera_model"] != config["camera_model"]:
        log_error("Configured camera matrix does not match the RAW camera model.")
    transform = build_color_transform(config["camera_coefficients"])
    print("Linearizing sensor samples...", flush=True)
    linear, _ = linearize(mosaic, metadata["black_level"], metadata["white_level"])
    stage_outputs = export_stage_outputs(
        mosaic, linear, metadata, config["pattern"], paths["run_dir"] / "stages",
        transform, config["target_mean"])
    del mosaic
    export_bayer_previews(
        linear, metadata["white_balance_gains"], paths["run_dir"] / "bayer_patterns",
        transform, config["target_mean"])
    print(f"White balance: {config['white_balance']}; candidate Bayer: {config['pattern']}",
          flush=True)
    if config.get("manual_patches"):
        preview_path = stage_outputs["white_balance"]["camera"]["exports"]["png"]["path"]
        with Image.open(preview_path) as image:
            image.thumbnail((1505, 1004))
            preview = np.asarray(image.convert("RGB"))
        compare_manual_balance(
            linear, config["pattern"], preview, config["target_mean"],
            lambda pattern, gains, target: render_image(linear, pattern, gains, transform, target),
            paths["run_dir"] / "manual_white_balance", config["manual_patches"])
    if "camera_path" in config:
        add_dcraw_comparison(config, stage_outputs, paths["run_dir"])
    print(f"Saved run: {paths['run_dir']}", flush=True)
    return paths["run_dir"]


def run_pipeline(config_path, log_path=None):
    from . import error_handling

    previous_log = error_handling.ERROR_LOG
    if log_path is not None:
        error_handling.ERROR_LOG = Path(log_path)
    try:
        try:
            main(config_path)
        except Exception as error:
            log_error(f"config={config_path}: {type(error).__name__}: {error}")
        return 0
    except SystemExit as stopped:
        return stopped.code
    finally:
        error_handling.ERROR_LOG = previous_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Develop RAW images and save sRGB comparisons.")
    parser.add_argument("config", type=Path, help="Path to the pipeline JSON configuration")
    arguments = parser.parse_args()
    sys.exit(run_pipeline(arguments.config))
