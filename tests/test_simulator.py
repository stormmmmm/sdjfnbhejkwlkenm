"""Проверки симулятора: физические свойства, фазовые PNG и полный экспорт."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from const import (
    IMAGE_SHAPE,
    IMAGE_SIZE,
    PHASE_MAX_RAD,
    PHASE_MIN_RAD,
    PIXEL_PITCH_M,
    RECONSTRUCTION_DISTANCES_M,
    WAVELENGTH_M,
)
from mathlib import output_pixel_pitch, reconstruct_intensity
from pictures import load_phase_png, phase_to_uint8, save_phase_png
from simulator import (
    export_results,
    generate_random_phase,
    phase_to_field,
    reconstruct_all,
    simulate,
)


class SimulatorPhysicsTests(unittest.TestCase):
    def test_zero_phase_has_finite_nonnegative_intensity_in_each_plane(self):
        result = simulate(np.zeros(IMAGE_SHAPE))
        self.assertEqual(result.field.shape, IMAGE_SHAPE)
        np.testing.assert_array_equal(result.field, np.ones(IMAGE_SHAPE))
        self.assertEqual(len(result.output_fields), 4)
        self.assertEqual(result.intensities.shape, (4, *IMAGE_SHAPE))
        for field, intensity in zip(result.output_fields, result.intensities):
            self.assertEqual(field.dtype, np.dtype(np.complex128))
            self.assertEqual(intensity.dtype, np.dtype(np.float64))
            self.assertTrue(np.isfinite(field).all())
            self.assertTrue(np.isfinite(intensity).all())
            self.assertTrue((intensity >= 0).all())
            np.testing.assert_array_equal(intensity, np.abs(field)**2)

    def test_random_phase_is_reproducible_and_covers_the_requested_range(self):
        phase = generate_random_phase(seed=83)
        self.assertEqual(phase.shape, IMAGE_SHAPE)
        self.assertEqual(phase.dtype, np.dtype(np.float64))
        self.assertTrue(np.isfinite(phase).all())
        self.assertTrue((phase >= PHASE_MIN_RAD).all())
        self.assertTrue((phase <= PHASE_MAX_RAD).all())
        np.testing.assert_array_equal(phase, generate_random_phase(seed=83))
        self.assertFalse(np.array_equal(phase, generate_random_phase(seed=84)))
        # Изменение возвращённого массива не влияет на следующие генерации.
        phase.fill(0)
        self.assertFalse(np.array_equal(phase, generate_random_phase(seed=83)))

    def test_same_input_field_generates_four_screens_without_mutating_phase(self):
        phase = generate_random_phase(seed=19)
        original = phase.copy()
        result = simulate(phase, seed=19, source="random")
        self.assertEqual(result.seed, 19)
        self.assertEqual(result.source, "random")
        self.assertEqual(result.intensities.shape, (4, *IMAGE_SHAPE))
        for index, distance in enumerate(RECONSTRUCTION_DISTANCES_M):
            expected = reconstruct_intensity(phase_to_field(original), distance)
            np.testing.assert_allclose(result.intensities[index], expected, rtol=1e-13, atol=0)
        np.testing.assert_array_equal(phase, original)
        phase.fill(0)
        np.testing.assert_array_equal(result.phase, original)

    def test_reconstruct_all_matches_simulation_and_uses_four_ffts(self):
        phase = generate_random_phase(seed=53)
        with patch("mathlib.np.fft.fft2", wraps=np.fft.fft2) as fft:
            intensities = reconstruct_all(phase)
        self.assertEqual(fft.call_count, 4)
        np.testing.assert_array_equal(intensities, simulate(phase).intensities)

    def test_global_phase_shift_does_not_change_any_screen_intensity(self):
        phase = generate_random_phase(seed=17)
        shifted = (phase + 0.73 + np.pi) % (2 * np.pi) - np.pi
        original = reconstruct_all(phase)
        translated = reconstruct_all(shifted)
        np.testing.assert_allclose(translated, original, rtol=2e-12, atol=2e-13)

    def test_discrete_power_is_preserved_on_each_distance_dependent_grid(self):
        phase = generate_random_phase(seed=61)
        intensities = reconstruct_all(phase)
        input_power = IMAGE_SIZE**2 * PIXEL_PITCH_M**2  # |H|² = 1.
        for distance, intensity in zip(RECONSTRUCTION_DISTANCES_M, intensities):
            with self.subTest(distance=distance):
                output_power = np.sum(intensity) * output_pixel_pitch(distance)**2
                np.testing.assert_allclose(output_power, input_power, rtol=2e-12, atol=0)

    def test_integer_linear_phase_ramp_shifts_intensity_in_expected_direction(self):
        # exp(+2πi*k*n/N) сдвигает спектр к положительному индексу k.
        # Столбцы задают x, строки — y. DFT-сдвиг циклический по границам.
        columns, rows = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))
        shift_x, shift_y = 3, -2
        ramp = 2 * np.pi * (shift_x * columns + shift_y * rows) / IMAGE_SIZE
        wrapped = (ramp + np.pi) % (2 * np.pi) - np.pi
        baseline = reconstruct_all(np.zeros(IMAGE_SHAPE))
        tilted = reconstruct_all(wrapped)
        expected = np.roll(baseline, shift=(shift_y, shift_x), axis=(1, 2))
        np.testing.assert_allclose(tilted, expected, rtol=3e-11, atol=3e-13)
        # Параксиальное смещение z*theta равно числу отсчётов * шаг экрана.
        theta_x = WAVELENGTH_M * shift_x / (IMAGE_SIZE * PIXEL_PITCH_M)
        for distance in RECONSTRUCTION_DISTANCES_M:
            self.assertAlmostEqual(
                distance * theta_x, shift_x * output_pixel_pitch(distance), places=16
            )

    def test_output_grid_steps_and_widths_use_si_units(self):
        expected_steps_um = (10.0165625, 10.5153125, 11.0140625, 11.5128125)
        expected_widths_mm = (1.28212, 1.34596, 1.40980, 1.47364)
        for distance, step_um, width_mm in zip(
            RECONSTRUCTION_DISTANCES_M, expected_steps_um, expected_widths_mm
        ):
            with self.subTest(distance=distance):
                actual = output_pixel_pitch(distance)
                self.assertAlmostEqual(actual * 1e6, step_um, places=10)
                self.assertAlmostEqual(IMAGE_SIZE * actual * 1e3, width_mm, places=10)

    def test_output_grid_rejects_nonpositive_or_nonfinite_distance(self):
        for distance in (0, -0.01, np.nan, np.inf, -np.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                output_pixel_pitch(distance)


class PhasePngTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "doe.png"

    def test_phase_png_endpoints_and_roundtrip_quantization(self):
        phase = np.linspace(PHASE_MIN_RAD, PHASE_MAX_RAD, IMAGE_SIZE**2).reshape(IMAGE_SHAPE)
        original = phase.copy()
        pixels = phase_to_uint8(phase)
        self.assertEqual(pixels.dtype, np.dtype(np.uint8))
        self.assertEqual(pixels[0, 0], 0)
        self.assertEqual(pixels[-1, -1], 255)
        save_phase_png(phase, self.path)
        with Image.open(self.path) as image:
            self.assertEqual((image.format, image.mode, image.size), ("PNG", "L", (128, 128)))
            np.testing.assert_array_equal(np.array(image), pixels)
        decoded = load_phase_png(self.path)
        self.assertEqual(decoded.dtype, np.dtype(np.float64))
        self.assertEqual(decoded[0, 0], PHASE_MIN_RAD)
        self.assertEqual(decoded[-1, -1], PHASE_MAX_RAD)
        self.assertLessEqual(np.max(np.abs(decoded - phase)), np.pi / 255 + 1e-14)
        np.testing.assert_array_equal(phase, original)

    def test_load_rejects_wrong_format_shape_and_pixel_modes(self):
        cases = (
            ("JPEG", "L", (128, 128)),
            ("PNG", "L", (127, 128)),
            ("PNG", "L", (128, 129)),
            ("PNG", "RGB", (128, 128)),
            ("PNG", "RGBA", (128, 128)),
            ("PNG", "P", (128, 128)),
            ("PNG", "1", (128, 128)),
            ("PNG", "I;16", (128, 128)),
        )
        for image_format, mode, size in cases:
            with self.subTest(format=image_format, mode=mode, size=size):
                Image.new(mode, size).save(self.path, format=image_format)
                with self.assertRaises(ValueError):
                    load_phase_png(self.path)

    def test_load_rejects_corrupt_png(self):
        self.path.write_bytes(b"not a png")
        with self.assertRaises(ValueError):
            load_phase_png(self.path)

    def test_save_preserves_existing_user_file(self):
        original = b"existing user file"
        self.path.write_bytes(original)
        with self.assertRaises(FileExistsError):
            save_phase_png(np.zeros(IMAGE_SHAPE), self.path)
        self.assertEqual(self.path.read_bytes(), original)


class SimulationExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run"

    def test_export_preserves_raw_intensity_and_reconstructs_saved_quantized_mask(self):
        phase = generate_random_phase(seed=123)
        result = simulate(phase, seed=123, source="random")
        original_intensities = result.intensities.copy()
        export_results(result, self.path)
        np.testing.assert_array_equal(np.load(self.path / "phase_radians.npy"), phase)
        metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["wavelength_m"], WAVELENGTH_M)
        self.assertEqual(metadata["n"], IMAGE_SIZE)
        self.assertEqual(metadata["pixel_pitch_m"], PIXEL_PITCH_M)
        self.assertEqual(metadata["distances_m"], list(RECONSTRUCTION_DISTANCES_M))
        self.assertEqual(metadata["seed"], 123)
        self.assertIn("normalization", metadata)
        np.testing.assert_allclose(
            metadata["output_pixel_pitches_m"],
            [output_pixel_pitch(distance) for distance in RECONSTRUCTION_DISTANCES_M],
            rtol=1e-15, atol=0,
        )
        quantized_phase = load_phase_png(self.path / "doe.png")
        quantized_intensities = reconstruct_all(quantized_phase)
        for index in range(4):
            name = str(index + 1)
            with self.subTest(plane=name):
                np.testing.assert_array_equal(
                    np.load(self.path / f"{name}.npy"), original_intensities[index]
                )
                np.testing.assert_allclose(
                    np.load(self.path / "quantized" / f"{name}.npy"),
                    quantized_intensities[index], rtol=1e-13, atol=0,
                )
                for folder, intensity in (
                    (self.path, original_intensities[index]),
                    (self.path / "quantized", quantized_intensities[index]),
                ):
                    with Image.open(folder / f"{name}.png") as image:
                        self.assertEqual(
                            (image.format, image.mode, image.size), ("PNG", "L", (128, 128))
                        )
                        expected = np.rint(255 * (intensity / intensity.max())).astype(np.uint8)
                        np.testing.assert_array_equal(np.array(image), expected)
        np.testing.assert_array_equal(result.intensities, original_intensities)

    def test_export_refuses_existing_directory_without_overwriting(self):
        self.path.mkdir()
        marker = self.path / "metadata.json"
        marker.write_text("existing user data", encoding="utf-8")
        result = simulate(np.zeros(IMAGE_SHAPE))
        with self.assertRaises(FileExistsError):
            export_results(result, self.path)
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing user data")
        self.assertEqual(list(self.path.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
