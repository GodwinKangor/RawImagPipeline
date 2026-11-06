# Author: Godwin Kangor
# Date: 2026-09-25
# File name: raw_conversion.py

from .error_handling import log_error

from pathlib import Path
import math
import re
import shlex
import shutil
import subprocess
import tempfile

import numpy as np
from skimage import io


# Keep reconnaissance and processing files in separate folders for each run.
def prepare_paths(raw_path, work_dir):
    source = Path(raw_path).resolve()
    workspace = Path(work_dir).resolve()
    if not source.is_file():
        log_error(f"RAW file not found: {source}")

    if source.suffix.lower() != ".cr2":
        log_error("Expected a .CR2 file.")

    if source.stat().st_size == 0:
        log_error("The RAW file is empty.")
    workspace.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=workspace))
    reconnaissance_dir = run_dir / "reconnaissance"
    working_dir = run_dir / "processing"
    reconnaissance_dir.mkdir()
    working_dir.mkdir()

    reconnaissance_raw = reconnaissance_dir / source.name
    working_raw = working_dir / source.name
    shutil.copy2(source, reconnaissance_raw)
    shutil.copy2(source, working_raw)

    return {
        "source_raw": source,
        "run_dir": run_dir,
        "reconnaissance_raw": reconnaissance_raw,
        "reconnaissance_dir": reconnaissance_dir,
        "working_raw": working_raw,
        "working_dir": working_dir,
    }

# shared helper function
def run_dcraw(raw_path, flags, log_name):
    source = Path(raw_path).resolve()
    if not source.is_file():
        log_error(f"RAW file not found: {source}")
    tiff_path = source.with_suffix(".tiff")
    if tiff_path.exists():
        log_error(f"Refusing to overwrite an existing TIFF: {tiff_path}")

    command = ["dcraw", *flags, str(source)]
    log_path = source.parent / log_name
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", cwd=source.parent, check=False)
    except FileNotFoundError as error:
        log_error("dcraw is not installed or is not on PATH.")

    output = result.stdout + "\n" + result.stderr
    log_path.write_text(
        f"Command: {shlex.join(command)}\nExit code: {result.returncode}\n{output}",
        encoding="utf-8")
    if result.returncode != 0:
        log_error(f"dcraw failed; see {log_path}")
    if not tiff_path.is_file() or tiff_path.stat().st_size == 0:
        log_error(f"dcraw did not produce a nonempty TIFF; see {log_path}")
    return output, tiff_path


# The reconnaissance TIFF is for metadata only, not image processing.
def extract_metadata(raw_path):
    output, _ = run_dcraw(raw_path, ["-4", "-d", "-v", "-w", "-T"],
                           "reconnaissance.log")
    log_path = Path(raw_path).resolve().parent / "reconnaissance.log"
    camera = re.search(r"Loading (.+?) image from ", output)
    scaling = re.search(
        r"Scaling with darkness\s+(\d+),\s+saturation\s+(\d+),\s+"
        r"and\s+multipliers\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", output)
    if camera is None or scaling is None:
        log_error(f"Required camera or scaling metadata missing; see {log_path}")

    black_level = int(scaling.group(1))
    white_level = int(scaling.group(2))
    gains = tuple(float(value) for value in scaling.groups()[2:])
    if white_level <= black_level:
        log_error(f"Invalid black/white levels; see {log_path}")
    if not all(math.isfinite(gain) and gain > 0 for gain in gains):
        log_error(f"White-balance gains must be finite and positive; see {log_path}")
    if not math.isclose(gains[1], gains[3], rel_tol=1e-5):
        log_error("Unequal green gains require a four-channel handling decision.")

    return {
        "camera_model": camera.group(1).strip(),
        "black_level": black_level,
        "white_level": white_level,
        "white_balance_gains": gains[:3],
    }


def convert_raw_to_tiff(raw_path):
    _, tiff_path = run_dcraw(raw_path, ["-4", "-D", "-T"], "conversion.log")
    return tiff_path


def load_mosaic(tiff_path):
    source = Path(tiff_path)
    if not source.is_file():
        log_error(f"Bayer TIFF not found: {source}")
    raw = io.imread(source)
    if raw.ndim != 2 or raw.size == 0:
        log_error("Expected a nonempty two-dimensional Bayer mosaic.")
    if not np.issubdtype(raw.dtype, np.unsignedinteger):
        log_error(f"Expected unsigned-integer samples, received {raw.dtype}.")

    image_info = {
        "height": raw.shape[0],
        "width": raw.shape[1],
        "original_dtype": str(raw.dtype),
        "storage_bits_per_pixel": raw.dtype.itemsize * 8,
    }
    # casting preserves sensor values; black-level normalization comes later.
    return raw.astype(np.float64), image_info


def prepare_raw(raw_path, work_dir):
    paths = prepare_paths(raw_path, work_dir)
    metadata = extract_metadata(paths["reconnaissance_raw"])
    tiff_path = convert_raw_to_tiff(paths["working_raw"])
    mosaic, image_info = load_mosaic(tiff_path)
    paths["tiff_path"] = tiff_path
    return mosaic, metadata, image_info, paths
