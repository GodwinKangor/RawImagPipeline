"""Small numerical checks; the full RAW run is performed separately."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from skimage import io

from raw_pipeline.pipeline import load_config, render_camera_rgb, export_images
from raw_pipeline.preprocessing import linearize, create_channel_masks
from raw_pipeline.white_balance import automatic_gains, manual_balance


class PipelineTests(unittest.TestCase):
    def test_sensor_to_rgb_all_patterns_and_methods(self):
        for pattern in ("RGGB", "GRBG", "BGGR", "GBRG"):
            masks = create_channel_masks((7, 9), pattern)
            sensor = np.empty((7, 9), dtype=np.uint16)
            for color, value in zip("RGB", (300, 500, 200)):
                sensor[masks[color]] = value
            linear, statistics = linearize(sensor, 100, 1100)
            self.assertEqual(statistics, {"black_fraction": 0, "saturation_fraction": 0})
            for method in ("white_world", "gray_world", "camera"):
                with self.subTest(pattern=pattern, method=method):
                    original = linear.copy()
                    gains = automatic_gains(linear, pattern, method, [2, 1, 4])
                    rgb = render_camera_rgb(linear, pattern, gains)
                    np.testing.assert_allclose(rgb, .4)
                    np.testing.assert_array_equal(linear, original)
                    self.assertEqual(rgb.shape, (7, 9, 3))
                    self.assertEqual(rgb.dtype, np.float64)
            gains, _ = manual_balance(linear, pattern, (1, 6, 1, 8))
            np.testing.assert_allclose(render_camera_rgb(linear, pattern, gains), .4)

    def test_measured_samples_and_interior_interpolation(self):
        rng = np.random.default_rng(4)
        for pattern in ("RGGB", "GRBG", "BGGR", "GBRG"):
            linear = rng.uniform(.01, 1, (7, 9))
            masks = create_channel_masks(linear.shape, pattern)
            rgb = render_camera_rgb(linear, pattern, [2, 1, 3])
            for index, (color, gain) in enumerate(zip("RGB", (2, 1, 3))):
                np.testing.assert_array_equal(rgb[:, :, index][masks[color]],
                                              linear[masks[color]] * gain)
            self.assertGreater(rgb.max(), 1)
            for row in range(1, 6):
                for column in range(1, 8):
                    if not masks["G"][row, column]:
                        expected = (linear[row-1, column] + linear[row+1, column]
                                    + linear[row, column-1] + linear[row, column+1]) / 4
                        self.assertAlmostEqual(rgb[row, column, 1], expected)

    def test_config_paths_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            config = {"raw_path": "input.CR2", "work_dir": "runs",
                      "pattern": "rggb", "white_balance": "camera",
                      "camera_model": "test camera", "coefficient_source": "synthetic fixture",
                      "camera_coefficients": [10000, 0, 0, 0, 10000, 0, 0, 0, 10000],
                      "target_mean": .25}
            path.write_text(json.dumps(config))
            loaded = load_config(path)
            self.assertEqual(loaded["raw_path"], (root / "input.CR2").resolve())
            self.assertEqual(loaded["pattern"], "RGGB")
            config["manual_patches"] = [{"label": "Neutral patch", "bounds": [1, 5, 1, 5]}]
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)["manual_patches"], config["manual_patches"])
            config["pattern"] = "invalid"
            path.write_text(json.dumps(config))
            with self.assertRaises(SystemExit):
                load_config(path)

    def test_export_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encoded = np.random.default_rng(3).uniform(0, 1, (8, 10, 3))
            records = export_images(encoded, root / "sample")
            np.testing.assert_array_equal(io.imread(records["png"]["path"]),
                                          np.rint(encoded * 255).astype(np.uint8))
            self.assertEqual(io.imread(records["jpeg"]["path"]).shape, encoded.shape)
            with self.assertRaises(SystemExit):
                export_images(encoded, root / "sample")
            with self.assertRaises(SystemExit):
                export_images(encoded * 2, root / "invalid")


if __name__ == "__main__":
    unittest.main()
