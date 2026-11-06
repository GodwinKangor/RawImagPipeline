"""Numerical checks for the assignment's color and display transformations."""

import unittest
import numpy as np

from raw_pipeline.color import build_color_transform, correct_color, adjust_brightness, gamma_encode_srgb
from raw_pipeline.pipeline import render_image


class ColorTests(unittest.TestCase):
    def test_transform_preserves_neutral(self):
        # Synthetic coefficients; these are not a camera calibration.
        transform = build_color_transform([10000, 0, 0, 0, 10000, 0, 0, 0, 10000])
        np.testing.assert_allclose(transform @ np.ones(3), np.ones(3))
        srgb_to_xyz = np.array([[.4124564, .3575761, .1804375],
                               [.2126729, .7151522, .0721750],
                               [.0193339, .1191920, .9503041]])
        forward = srgb_to_xyz / srgb_to_xyz.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(transform @ forward, np.eye(3), atol=1e-14)

    def test_color_orientation_and_no_clipping(self):
        source = np.array([[[1., 2., 3.]]])
        matrix = np.array([[1., -2., 0.], [0., 3., 0.], [0., 0., 2.]])
        np.testing.assert_array_equal(correct_color(source, matrix), [[[-3., 6., 6.]]])
        np.testing.assert_array_equal(source, [[[1., 2., 3.]]])

    def test_brightness_and_clipping(self):
        source = np.full((2, 3, 3), .1)
        bright, stats = adjust_brightness(source, .25)
        np.testing.assert_allclose(bright, .25)
        self.assertAlmostEqual(stats['scale'], 2.5)
        self.assertAlmostEqual(stats['achieved_mean'], .25)
        np.testing.assert_array_equal(source, .1)
        source = np.zeros((10, 10, 3))
        source[0, 0] = 1
        source[1, 1] = -1
        bright, stats = adjust_brightness(source, .25)
        self.assertAlmostEqual(stats['clipped_pixel_fraction'], .01)
        self.assertAlmostEqual(stats['negative_pixel_fraction'], .01)
        self.assertLess(stats['achieved_mean'], .25)
        self.assertTrue(np.all((bright >= 0) & (bright <= 1)))

    def test_transfer_curve(self):
        linear = np.array([[[0., .0031308, 1.], [.18, .5, .0031308 + 1e-10]]])
        encoded = gamma_encode_srgb(linear)
        self.assertEqual(encoded[0, 0, 0], 0)
        self.assertAlmostEqual(encoded[0, 0, 2], 1)
        self.assertAlmostEqual(encoded[0, 0, 1], 12.92 * .0031308)
        self.assertAlmostEqual(encoded[0, 1, 0], .4613561295)
        self.assertLess(abs(encoded[0, 1, 2] - encoded[0, 0, 1]), 1e-7)

    def test_render_image_integration(self):
        encoded, stats = render_image(np.full((6, 8), .2), 'RGGB', [1, 1, 1],
                                      np.eye(3), .25)
        np.testing.assert_allclose(encoded, 1.055 * .25 ** (1 / 2.4) - .055)
        self.assertAlmostEqual(stats['achieved_mean'], .25)

    def test_invalid_inputs(self):
        calls = [lambda: build_color_transform([0] * 9),
                 lambda: build_color_transform([10000] * 9),
                 lambda: correct_color(np.ones((2, 2)), np.eye(3)),
                 lambda: adjust_brightness(np.zeros((2, 2, 3)), .25),
                 lambda: adjust_brightness(np.ones((2, 2, 3)), 0),
                 lambda: gamma_encode_srgb(np.full((2, 2, 3), 1.1))]
        for call in calls:
            with self.assertRaises(SystemExit):
                call()


if __name__ == '__main__':
    unittest.main()
