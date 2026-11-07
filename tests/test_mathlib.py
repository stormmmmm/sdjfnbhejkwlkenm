import cmath
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from const import (
    IMAGE_SHAPE,
    IMAGE_SIZE,
    MAX_PIXEL_VALUE,
    PHASE_MAX_RAD,
    PHASE_MIN_RAD,
    PIXEL_PITCH_M,
    RECONSTRUCTION_DISTANCES_M,
    WAVELENGTH_M,
    WAVE_NUMBER_PER_M,
)
from mathlib import (
    fresnel_field_matrix,
    fresnel_input,
    fresnel_propagate,
    reconstruct_intensity,
    slm_field,
    slm_field_matrix,
)


class SLMFieldTests(unittest.TestCase):
    def test_known_phases_and_range_endpoints(self):
        samples = ((-np.pi, -1), (-np.pi / 2, -1j), (0, 1), (np.pi / 2, 1j), (np.pi, -1))
        for phase, expected in samples:
            with self.subTest(phase=phase):
                field = slm_field(phase)
                self.assertIsInstance(field, complex)
                self.assertAlmostEqual(field, expected, places=14)

    def test_amplitude_sets_field_magnitude_and_intensity(self):
        for amplitude in (0.0, 0.3, 2.0):
            with self.subTest(amplitude=amplitude):
                field = slm_field(0.7, amplitude)
                self.assertAlmostEqual(abs(field), amplitude)
                self.assertAlmostEqual(abs(field) ** 2, amplitude**2)

    def test_accepts_numpy_numeric_scalars(self):
        result = slm_field(np.float32(0.5), np.float64(2.0))
        self.assertAlmostEqual(result, 2 * cmath.exp(0.5j))

    def test_rejects_invalid_phase_values(self):
        for value in (np.nan, np.inf, -np.inf, -np.pi - 0.01, np.pi + 0.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                slm_field(value)

    def test_rejects_invalid_amplitudes(self):
        for value in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                slm_field(0, value)

class SLMFieldMatrixTests(unittest.TestCase):
    def test_known_phases_preserve_pixel_positions_and_input(self):
        phases = np.zeros(IMAGE_SHAPE)
        phases[0, 0] = PHASE_MIN_RAD
        phases[3, 7] = np.pi / 2
        phases[7, 3] = -np.pi / 2
        phases[-1, -1] = PHASE_MAX_RAD
        original = phases.copy()
        expected = np.ones(IMAGE_SHAPE, dtype=np.complex128)
        expected[0, 0] = expected[-1, -1] = -1
        expected[3, 7], expected[7, 3] = 1j, -1j

        field = slm_field_matrix(phases)

        self.assertEqual(field.shape, IMAGE_SHAPE)
        self.assertEqual(field.dtype, np.dtype(np.complex128))
        self.assertTrue(np.isfinite(field).all())
        np.testing.assert_allclose(field, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(phases, original)
        field.fill(0)
        np.testing.assert_array_equal(phases, original)

    def test_uniform_amplitude_sets_intensity_across_whole_field(self):
        phases = np.linspace(-3, 3, IMAGE_SIZE**2).reshape(IMAGE_SHAPE).astype(np.float32)
        for amplitude in (0.0, 0.3, 2.0):
            with self.subTest(amplitude=amplitude):
                field = slm_field_matrix(phases, amplitude)
                np.testing.assert_allclose(np.abs(field)**2, amplitude**2, rtol=1e-14, atol=0)

    def test_rejects_wrong_shape(self):
        for shape in ((), (IMAGE_SIZE,), (127, 128), (128, 127), (4, 128, 128), (128, 128, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                slm_field_matrix(np.zeros(shape))

    def test_rejects_invalid_phase_values_anywhere_in_matrix(self):
        for value in (np.nan, np.inf, -np.inf, PHASE_MIN_RAD - 0.01, PHASE_MAX_RAD + 0.01):
            for pixel in ((0, 0), (17, 93), (127, 127)):
                phases = np.zeros(IMAGE_SHAPE)
                phases[pixel] = value
                with self.subTest(value=value, pixel=pixel), self.assertRaises(ValueError):
                    slm_field_matrix(phases)

    def test_rejects_non_real_numeric_phase_types(self):
        for dtype in (complex, str, object, bool):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                slm_field_matrix(np.zeros(IMAGE_SHAPE, dtype=dtype))

    def test_rejects_invalid_amplitude(self):
        phases = np.zeros(IMAGE_SHAPE)
        for amplitude in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(amplitude=amplitude), self.assertRaises(ValueError):
                slm_field_matrix(phases, amplitude)


class FresnelPixelTests(unittest.TestCase):
    def test_input_at_origin_equals_original_field(self):
        field = 0.3 + 0.7j
        result = fresnel_input(field, 0, 0, RECONSTRUCTION_DISTANCES_M[0])
        self.assertIsInstance(result, complex)
        self.assertEqual(result, field)

    def test_input_quadratic_phase_rotates_field_without_changing_amplitude(self):
        field = 0.3 + 0.7j
        for distance in RECONSTRUCTION_DISTANCES_M:
            with self.subTest(distance=distance):
                u = np.sqrt(WAVELENGTH_M * distance / 2)
                result = fresnel_input(field, u, 0, distance)
                self.assertAlmostEqual(result, 1j * field, places=13)
                self.assertAlmostEqual(abs(result), abs(field))

    def test_output_at_origin_includes_prefactor_and_input_pixel_area(self):
        fft_value = 3 - 2j
        for distance in RECONSTRUCTION_DISTANCES_M:
            with self.subTest(distance=distance):
                result = fresnel_propagate(fft_value, 0, 0, distance)
                expected = (
                    cmath.exp(1j * WAVE_NUMBER_PER_M * distance)
                    / (1j * WAVELENGTH_M * distance) * fft_value * PIXEL_PITCH_M**2
                )
                self.assertIsInstance(result, complex)
                self.assertAlmostEqual(result, expected, places=14)

    def test_output_quadratic_phase_rotates_origin_value(self):
        distance = RECONSTRUCTION_DISTANCES_M[0]
        y = np.sqrt(WAVELENGTH_M * distance / 2)
        at_origin = fresnel_propagate(2 + 1j, 0, 0, distance)
        self.assertAlmostEqual(fresnel_propagate(2 + 1j, 0, y, distance), 1j * at_origin, places=13)

    def test_zero_input_and_fft_coefficient_give_zero(self):
        distance = RECONSTRUCTION_DISTANCES_M[0]
        self.assertEqual(fresnel_input(0, 10e-6, 20e-6, distance), 0j)
        self.assertEqual(fresnel_propagate(0, 10e-6, 20e-6, distance), 0j)

    def test_rejects_invalid_distances(self):
        for function in (fresnel_input, fresnel_propagate):
            for distance in (0, -0.0241, np.nan, np.inf, -np.inf):
                with self.subTest(function=function.__name__, distance=distance):
                    with self.assertRaises(ValueError):
                        function(1, 0, 0, distance)

class FresnelIntegrationTests(unittest.TestCase):
    """Сравнение распространения с интегралом и проверка сохранённого ДОЭ."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.field = rng.uniform(0.2, 1.0, IMAGE_SHAPE) * np.exp(
            1j * rng.uniform(PHASE_MIN_RAD, PHASE_MAX_RAD, IMAGE_SHAPE)
        )
        self.axis = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * PIXEL_PITCH_M

    def test_scalar_operations_with_one_fft_match_direct_integral(self):
        # Скалярные входные множители, один общий FFT, скалярные выходы.
        u, v = np.meshgrid(self.axis, self.axis)
        for distance in RECONSTRUCTION_DISTANCES_M:
            prepared = np.array([
                [fresnel_input(self.field[row, col], self.axis[col], self.axis[row], distance)
                 for col in range(IMAGE_SIZE)]
                for row in range(IMAGE_SIZE)
            ])
            spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(prepared)))
            pitch = WAVELENGTH_M * distance / (IMAGE_SIZE * PIXEL_PITCH_M)
            for row, col in ((0, 0), (64, 64), (65, 60), (127, 127), (15, 80)):
                with self.subTest(distance=distance, pixel=(row, col)):
                    x, y = (col - IMAGE_SIZE // 2) * pitch, (row - IMAGE_SIZE // 2) * pitch
                    result = fresnel_propagate(spectrum[row, col], x, y, distance)
                    kernel = np.exp(
                        1j * np.pi * ((x - u)**2 + (y - v)**2) / (WAVELENGTH_M * distance)
                    )
                    expected = (
                        np.exp(1j * WAVE_NUMBER_PER_M * distance) / (1j * WAVELENGTH_M * distance)
                        * np.sum(self.field * kernel) * PIXEL_PITCH_M**2
                    )
                    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-11)

    def test_power_conservation_for_scalar_output_pixels(self):
        distance = RECONSTRUCTION_DISTANCES_M[0]
        prepared = self.field * np.exp(
            1j * np.pi * (self.axis[:, None]**2 + self.axis[None, :]**2)
            / (WAVELENGTH_M * distance)
        )
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(prepared)))
        pitch = WAVELENGTH_M * distance / (IMAGE_SIZE * PIXEL_PITCH_M)
        output_axis = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * pitch
        output_power = sum(
            abs(fresnel_propagate(spectrum[row, col], output_axis[col], output_axis[row], distance))**2
            for row in range(IMAGE_SIZE) for col in range(IMAGE_SIZE)
        ) * pitch**2
        input_power = np.sum(np.abs(self.field)**2) * PIXEL_PITCH_M**2
        np.testing.assert_allclose(output_power, input_power, rtol=1e-12, atol=0)

    def test_saved_quantized_doe_supplies_pixel_phase_for_all_four_planes(self):
        # Только тестовая маска. Нейросетевой синтез здесь не выполняется.
        phase = np.linspace(PHASE_MIN_RAD, PHASE_MAX_RAD, IMAGE_SIZE**2).reshape(IMAGE_SHAPE)
        encoded = np.clip(
            np.rint(MAX_PIXEL_VALUE * (phase - PHASE_MIN_RAD) / (PHASE_MAX_RAD - PHASE_MIN_RAD)),
            0, MAX_PIXEL_VALUE,
        ).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doe.png"
            Image.fromarray(encoded).save(path)
            with Image.open(path) as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "L", (128, 128)))
                pixels = np.array(image, dtype=np.float64)
            decoded = pixels / MAX_PIXEL_VALUE * (PHASE_MAX_RAD - PHASE_MIN_RAD) + PHASE_MIN_RAD
            self.assertEqual(decoded[0, 0], PHASE_MIN_RAD)
            self.assertEqual(decoded[-1, -1], PHASE_MAX_RAD)
            half_step = (PHASE_MAX_RAD - PHASE_MIN_RAD) / (2 * MAX_PIXEL_VALUE)
            self.assertLessEqual(np.max(np.abs(decoded - phase)), half_step + 1e-14)
            field = slm_field_matrix(decoded)
            original = field.copy()
            for distance in RECONSTRUCTION_DISTANCES_M:
                with self.subTest(distance=distance):
                    intensity = reconstruct_intensity(field, distance)
                    self.assertEqual(intensity.shape, IMAGE_SHAPE)
                    self.assertEqual(intensity.dtype, np.dtype(np.float64))
                    self.assertTrue(np.isfinite(intensity).all())
                    self.assertTrue((intensity >= 0).all())
                    np.testing.assert_array_equal(field, original)


class FresnelReconstructionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.field = rng.uniform(0.2, 1.0, IMAGE_SHAPE) * np.exp(
            1j * rng.uniform(PHASE_MIN_RAD, PHASE_MAX_RAD, IMAGE_SHAPE)
        )

    def test_output_field_matches_direct_integral_in_all_four_planes(self):
        axis = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * PIXEL_PITCH_M
        u, v = np.meshgrid(axis, axis)
        for distance in RECONSTRUCTION_DISTANCES_M:
            result = fresnel_field_matrix(self.field, distance)
            self.assertEqual(result.shape, IMAGE_SHAPE)
            self.assertEqual(result.dtype, np.dtype(np.complex128))
            pitch = WAVELENGTH_M * distance / (IMAGE_SIZE * PIXEL_PITCH_M)
            for row, col in ((0, 0), (64, 64), (65, 60), (127, 127), (15, 80)):
                with self.subTest(distance=distance, pixel=(row, col)):
                    x, y = (col - 64) * pitch, (row - 64) * pitch
                    kernel = np.exp(
                        1j * np.pi * ((x - u)**2 + (y - v)**2) / (WAVELENGTH_M * distance)
                    )
                    expected = (
                        np.exp(1j * WAVE_NUMBER_PER_M * distance) / (1j * WAVELENGTH_M * distance)
                        * np.sum(self.field * kernel) * PIXEL_PITCH_M**2
                    )
                    np.testing.assert_allclose(result[row, col], expected, rtol=1e-10, atol=1e-11)

    def test_intensity_preserves_power_and_input_for_all_four_distances(self):
        original = self.field.copy()
        input_power = np.sum(np.abs(self.field)**2) * PIXEL_PITCH_M**2
        for distance in RECONSTRUCTION_DISTANCES_M:
            with self.subTest(distance=distance):
                intensity = reconstruct_intensity(self.field, distance)
                pitch = WAVELENGTH_M * distance / (IMAGE_SIZE * PIXEL_PITCH_M)
                self.assertEqual(intensity.shape, IMAGE_SHAPE)
                self.assertEqual(intensity.dtype, np.dtype(np.float64))
                self.assertTrue(np.isfinite(intensity).all())
                self.assertTrue((intensity >= 0).all())
                np.testing.assert_allclose(np.sum(intensity) * pitch**2, input_power, rtol=1e-12, atol=0)
                np.testing.assert_array_equal(self.field, original)

    def test_single_input_pixel_gives_analytical_unnormalized_intensity_everywhere(self):
        field = np.zeros(IMAGE_SHAPE, dtype=np.complex128)
        field[17, 93] = 0.3 + 0.7j
        for distance in RECONSTRUCTION_DISTANCES_M:
            with self.subTest(distance=distance):
                intensity = reconstruct_intensity(field, distance)
                expected = abs(field[17, 93])**2 * PIXEL_PITCH_M**4 / (WAVELENGTH_M * distance)**2
                np.testing.assert_allclose(intensity, expected, rtol=1e-12, atol=0)

    def test_zero_field_returns_zero_intensity(self):
        intensity = reconstruct_intensity(np.zeros(IMAGE_SHAPE), RECONSTRUCTION_DISTANCES_M[0])
        self.assertEqual(intensity.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(intensity, np.zeros(IMAGE_SHAPE))

    def test_exactly_one_fft_is_used_per_plane(self):
        for distance in RECONSTRUCTION_DISTANCES_M:
            with self.subTest(distance=distance), patch("mathlib.np.fft.fft2", wraps=np.fft.fft2) as fft:
                reconstruct_intensity(self.field, distance)
                self.assertEqual(fft.call_count, 1)

    def test_rejects_invalid_fields(self):
        invalid_fields = [np.zeros(shape) for shape in ((), (128,), (127, 128), (4, 128, 128))]
        invalid_fields.extend(np.zeros(IMAGE_SHAPE, dtype=dtype) for dtype in (str, object, bool))
        for value in (np.nan, np.inf, -np.inf, complex(1, np.nan), complex(0, np.inf)):
            field = self.field.copy()
            field[-1, 0] = value
            invalid_fields.append(field)
        for function in (fresnel_field_matrix, reconstruct_intensity):
            for index, field in enumerate(invalid_fields):
                with self.subTest(function=function.__name__, case=index), self.assertRaises(ValueError):
                    function(field, RECONSTRUCTION_DISTANCES_M[0])

    def test_new_matrix_functions_reject_invalid_distances(self):
        for function in (fresnel_field_matrix, reconstruct_intensity):
            for distance in (0, -0.0241, np.nan, np.inf, -np.inf):
                with self.subTest(function=function.__name__, distance=distance), self.assertRaises(ValueError):
                    function(self.field, distance)


if __name__ == "__main__":
    unittest.main()
