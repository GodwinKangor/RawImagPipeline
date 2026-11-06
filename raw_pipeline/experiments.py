# Author: Godwin Kangor
# Date: 2026-09-26
# File name: experiments.py

from .error_handling import log_error

import json
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np

from PIL import Image, ImageDraw, ImageOps
from skimage import io

from .color import adjust_brightness, correct_color, gamma_encode_srgb
from .white_balance import manual_balance, select_neutral_patch


# Use shared exposure and camera gains; resize only after demosaicing.
def export_bayer_previews(linear_mosaic, camera_gains, output_dir, camera_to_srgb,
                          target_mean):
    from .pipeline import downsample, render_camera_rgb, to_uint8
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("RGGB", "GRBG", "BGGR", "GBRG")
    for name in (*patterns, "comparison"):
        if (output_dir / f"{name}.png").exists():
            log_error(f"Preview already exists: {name}.png")
    records = {}
    thumbnails = []
    scale = None
    for pattern in patterns:
        print(f"Generating Bayer preview: {pattern}", flush=True)
        preview = downsample(render_camera_rgb(linear_mosaic, pattern, camera_gains))
        preview = correct_color(preview, camera_to_srgb)
        if scale is None:
            _, measurements = adjust_brightness(preview, target_mean)
            scale = measurements["scale"]
        pixels = to_uint8(gamma_encode_srgb(np.clip(preview * scale, 0, 1)))
        path = output_dir / f"{pattern}.png"
        io.imsave(path, pixels, check_contrast=False)
        records[pattern] = str(path.resolve())
        thumbnail = Image.fromarray(pixels)
        thumbnail.thumbnail((720, 480))
        thumbnails.append((pattern, thumbnail))

    comparison = Image.new("RGB", (1440, 1080), "white")
    draw = ImageDraw.Draw(comparison)
    draw.text((15, 10), "Bayer candidates - color-corrected sRGB, shared exposure",
              fill="black")
    for index, (pattern, thumbnail) in enumerate(thumbnails):
        x, y = (index % 2) * 720, 40 + (index // 2) * 520
        draw.text((x + 15, y), pattern, fill="black")
        comparison.paste(thumbnail, (x, y + 25))
    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)
    records["comparison"] = str(comparison_path.resolve())
    records["settings"] = {"camera_gains": list(map(float, camera_gains)),
                           "brightness_scale": scale, "downsample_step": 4,
                           "color_correction": True,
                           "camera_to_srgb": camera_to_srgb.tolist(),
                           "note": "sRGB previews; exposure based on RGGB candidate"}
    (output_dir / "settings.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


# Saved bounds use full-resolution rows/columns; the preview is only for selection.
def compare_manual_balance(linear_mosaic, pattern, preview, target_mean,
                           render, output_dir, patches=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    preview = np.asarray(preview)
    if patches is None:
        patches = [{"label": f"Patch {index + 1}",
                    "bounds": select_neutral_patch(preview, linear_mosaic.shape)}
                   for index in range(3)]
    if not patches:
        log_error("Select at least one neutral patch.")
    annotated = Image.fromarray(preview).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    overview = Image.new("RGB", (600 * len(patches), 440), "white")
    labels = ImageDraw.Draw(overview)
    height, width = linear_mosaic.shape
    sx, sy = annotated.width / width, annotated.height / height
    records = []
    for index, patch in enumerate(patches):
        gains, measurements = manual_balance(linear_mosaic, pattern, patch["bounds"])
        print(f"Manual patch {index + 1}: {patch['label']}; gains {gains}", flush=True)
        encoded, brightness = render(pattern, gains, target_mean)
        image = Image.fromarray(np.rint(encoded * 255).astype(np.uint8))
        del encoded
        path = output_dir / f"patch_{index + 1}.png"
        image.save(path)
        image.thumbnail((600, 400))
        overview.paste(image, (index * 600, 40))
        labels.text((index * 600 + 10, 10), patch["label"], fill="black")
        r0, r1, c0, c1 = measurements["patch_bounds"]
        draw.rectangle((c0 * sx, r0 * sy, c1 * sx, r1 * sy), outline="red", width=2)
        draw.text((c0 * sx, r0 * sy - 15), str(index + 1), fill="red")
        records.append({"label": patch["label"], "gains": gains.tolist(),
                        **measurements, "brightness": brightness, "png": str(path)})
    annotated.save(output_dir / "selected_patches.png")
    overview.save(output_dir / "comparison.png")
    result = {"pattern": pattern, "target_mean": target_mean,
              "coordinate_order": "row_start, row_stop, column_start, column_stop; zero-based, exclusive stops",
              "note": "Candidate neutral surfaces selected visually; actual reflectance is unverified.",
              "patches": records}
    (output_dir / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def render_dcraw_reference(raw_path, flags, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(raw_path).resolve()
    working_raw = output_dir / source.name
    shutil.copy2(source, working_raw)
    command = ["dcraw", *flags, str(working_raw)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log = output_dir / "dcraw.log"
    log.write_text(f"Command: {shlex.join(command)}\nExit code: {result.returncode}\n"
                   f"{result.stdout}\n{result.stderr}")
    command_path = output_dir / "command.json"
    command_path.write_text(json.dumps(command, indent=2) + "\n")
    if result.returncode != 0:
        log_error(f"dcraw rendering failed; see {log}")
    ppm = working_raw.with_suffix(".ppm")
    png = output_dir / "dcraw_pipeline.png"
    # The reference command outputs 8-bit encoded RGB; only change the container.
    with Image.open(ppm) as image:
        if image.mode != "RGB":
            log_error("Expected an 8-bit RGB dcraw reference.")
        image.save(png)
    return {"png": png, "ppm": ppm, "command": command_path, "log": log}


# Compare full frames; account for crop differences before comparing individual pixels.
def compare_renderings(custom_path, dcraw_path, camera_path, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {"our_pipeline.png": Path(custom_path),
               "dcraw_pipeline.png": Path(dcraw_path),
               "camera_original.jpg": Path(camera_path)}
    labels = ("Our pipeline", "dcraw pipeline", "Camera JPEG")
    overview = Image.new("RGB", (2160, 560), "white")
    draw = ImageDraw.Draw(overview)
    records = {}
    for index, ((name, source), label) in enumerate(zip(sources.items(), labels)):
        destination = output_dir / name
        shutil.copy2(source, destination)
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        records[label] = {"path": str(destination), "size": list(image.size)}
        if label == "Camera JPEG":
            image.save(output_dir / "camera_decoded.png")
        image.thumbnail((720, 480))
        x = index * 720
        draw.text((x + 15, 15), label, fill="black")
        overview.paste(image, (x, 45))
    overview_path = output_dir / "comparison.png"
    overview.save(overview_path)
    records["note"] = ("Full-frame visual comparison; views resized independently. "
                       "Camera crop may differ; images are not registered for pixelwise comparison. "
                       "camera_original.jpg is unchanged; camera_decoded.png adds no detail.")
    (output_dir / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    return {"comparison": overview_path, "records": records}


# Render a dcraw reference with matching settings and compare it to own and the camera's.
def add_dcraw_comparison(config, stage_outputs, run_dir):
    method = config["white_balance"]
    settings = stage_outputs["white_balance"][method]
    flags = ["-v", "-q", "0", "-o", "1", "-g", "2.4", "12.92", "-W",
             "-b", str(settings["brightness"]["scale"])]
    if method == "camera":
        flags += ["-w"]
    else:
        red, green, blue = settings["gains"]
        flags += ["-r", str(red), str(green), str(blue), str(green)]
    reference = render_dcraw_reference(config["raw_path"], flags, run_dir / "dcraw_reference")
    comparison = compare_renderings(settings["exports"]["png"]["path"], reference["png"],
                                    config["camera_path"], run_dir / "final_comparison")
    return {"dcraw_reference": reference, "final_comparison": comparison}
