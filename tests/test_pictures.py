import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from const import RECONSTRUCTION_DISTANCES_M
from mathlib import reconstruct_intensity, slm_field_matrix
from pictures import InputPictures, OutputPictures


class InputPicturesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.pictures = InputPictures(str(self.path))

    def write_valid_inputs(self):
        # Создаём в обратном порядке, чтобы проверить сортировку имён.
        for number, value in reversed(list(enumerate((0, 64, 128, 255), start=1))):
            Image.new("L", (128, 128), color=value).save(self.path / f"{number}.png")

    def test_load_and_to_numpy_preserve_pixels_and_file_order(self):
        self.write_valid_inputs()
        self.pictures.load()
        result = self.pictures.to_numpy()

        self.assertEqual(result.shape, (4, 128, 128))
        self.assertEqual(result.dtype, np.dtype(np.uint8))
        for index, value in enumerate((0, 64, 128, 255)):
            np.testing.assert_array_equal(result[index], np.full((128, 128), value))

        result.fill(12)
        self.assertEqual(self.pictures.images[0][0, 0], 0)

    def test_normalize_is_float32_and_preserves_originals(self):
        self.write_valid_inputs()
        self.pictures.load()
        original = self.pictures.to_numpy()

        result = self.pictures.normalize()

        self.assertEqual(result.shape, (4, 128, 128))
        self.assertEqual(result.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.min(), 0.0)
        self.assertEqual(result.max(), 1.0)
        np.testing.assert_allclose(result[:, 0, 0], [0, 64 / 255, 128 / 255, 1])
        np.testing.assert_array_equal(self.pictures.to_numpy(), original)
        result.fill(0)
        np.testing.assert_array_equal(self.pictures.to_numpy(), original)

    def test_wrong_file_count_raises_type_error(self):
        for count in (0, 1, 2, 3, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                for number in range(count):
                    (Path(directory) / f"{number}.txt").write_text("test", encoding="utf-8")
                with self.assertRaises(TypeError):
                    InputPictures(directory).load()

    def test_all_files_are_counted_including_non_images(self):
        self.write_valid_inputs()
        (self.path / "extra.txt").write_text("test", encoding="utf-8")
        with self.assertRaises(TypeError):
            self.pictures.load()

    def test_subdirectories_are_not_input_files(self):
        self.write_valid_inputs()
        (self.path / "subdirectory").mkdir()
        self.pictures.load()
        self.assertEqual(self.pictures.to_numpy().shape, (4, 128, 128))

    def test_rejects_invalid_format_size_and_channels(self):
        invalid_images = (
            ("JPEG", "L", (128, 128)),
            ("PNG", "L", (64, 128)),
            ("PNG", "RGB", (128, 128)),
            ("PNG", "RGBA", (128, 128)),
            ("PNG", "P", (128, 128)),
            ("PNG", "1", (128, 128)),
            ("PNG", "I;16", (128, 128)),
        )
        for image_format, mode, size in invalid_images:
            with self.subTest(format=image_format, mode=mode, size=size):
                self.write_valid_inputs()
                Image.new(mode, size).save(self.path / "1.png", format=image_format)
                with self.assertRaises(ValueError):
                    self.pictures.load()

    def test_rejects_unreadable_file(self):
        self.write_valid_inputs()
        (self.path / "1.png").write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            self.pictures.load()

    def test_requires_loading_before_conversion_or_normalization(self):
        for method in (self.pictures.to_numpy, self.pictures.normalize):
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method()

    def test_failed_reload_preserves_previously_loaded_images(self):
        self.write_valid_inputs()
        self.pictures.load()
        original = self.pictures.to_numpy()
        Image.new("RGB", (128, 128)).save(self.path / "4.png")
        with self.assertRaises(ValueError):
            self.pictures.load()
        np.testing.assert_array_equal(self.pictures.to_numpy(), original)

    def test_missing_directory_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            InputPictures(self.path / "missing").load()


class OutputPicturesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "output"

    def test_constructor_copies_intensities_without_changing_original(self):
        intensities = np.full((4, 128, 128), 0.25, dtype=np.float32)
        original = intensities.copy()

        pictures = OutputPictures(str(self.path), intensities)

        self.assertEqual(pictures.path_output, self.path)
        self.assertEqual(pictures.intensities.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(intensities, original)
        intensities.fill(10)
        np.testing.assert_array_equal(pictures.intensities, original)
        self.assertFalse(self.path.exists())

    def test_rejects_wrong_intensity_shape(self):
        for shape in ((), (128, 128), (3, 128, 128), (5, 128, 128), (4, 127, 128), (4, 128, 128, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                OutputPictures(self.path, np.zeros(shape))

    def test_rejects_non_real_numeric_intensity_types(self):
        for dtype in (complex, str, object, bool):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                OutputPictures(self.path, np.zeros((4, 128, 128), dtype=dtype))

    def test_rejects_negative_or_nonfinite_intensities(self):
        for value in (-0.1, np.nan, np.inf, -np.inf):
            intensities = np.zeros((4, 128, 128))
            intensities[-1, -1, -1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                OutputPictures(self.path, intensities)

    def test_save_creates_directory_and_preserves_png_format_and_plane_order(self):
        intensities = np.zeros((4, 128, 128))
        expected = np.zeros((4, 128, 128), dtype=np.uint8)
        for index, peak in enumerate((0.01, 2.0, 30.0, 1e6)):
            intensities[index, index, :4] = peak * np.array([0.0, 0.25, 0.5, 1.0])
            expected[index, index, :4] = [0, 64, 128, 255]
        path_output = self.path / "nested" / "reconstructions"

        paths = OutputPictures(path_output, intensities).save()

        self.assertEqual(paths, [path_output / f"{index}.png" for index in range(1, 5)])
        self.assertEqual(len(list(path_output.iterdir())), 4)
        for index, path in enumerate(paths):
            with self.subTest(path=path), Image.open(path) as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "L", (128, 128)))
                np.testing.assert_array_equal(np.array(image), expected[index])

    def test_normalization_handles_zero_constant_and_large_intensities(self):
        intensities = np.zeros((4, 128, 128))
        intensities[1] = 7.0
        intensities[2] = 1e-300
        intensities[3] = np.finfo(np.float64).max
        intensities[3, 0, 0] /= 2
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            pixels = OutputPictures(self.path, intensities).to_uint8()

        self.assertEqual(pixels.shape, (4, 128, 128))
        self.assertEqual(pixels.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(pixels[0], np.zeros((128, 128), dtype=np.uint8))
        np.testing.assert_array_equal(pixels[1:3], np.full((2, 128, 128), 255, dtype=np.uint8))
        self.assertEqual(pixels[3, 0, 0], 128)
        self.assertEqual(pixels[3, -1, -1], 255)

    def test_processing_and_saving_preserve_original_intensities(self):
        intensities = np.linspace(0, 10, 4 * 128 * 128).reshape(4, 128, 128)
        original = intensities.copy()
        pictures = OutputPictures(self.path, intensities)

        pixels = pictures.to_uint8()
        pixels.fill(0)
        pictures.save()

        np.testing.assert_array_equal(intensities, original)
        np.testing.assert_array_equal(pictures.intensities, original)

    def test_existing_result_is_preserved_before_any_files_are_written(self):
        self.path.mkdir()
        existing = self.path / "3.png"
        existing.write_bytes(b"existing user data")
        pictures = OutputPictures(self.path, np.ones((4, 128, 128)))

        with self.assertRaises(FileExistsError):
            pictures.save()

        self.assertEqual(existing.read_bytes(), b"existing user data")
        self.assertEqual(list(self.path.iterdir()), [existing])

    def test_destination_file_is_preserved(self):
        self.path.write_text("existing user data", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            OutputPictures(self.path, np.ones((4, 128, 128))).save()

        self.assertEqual(self.path.read_text(encoding="utf-8"), "existing user data")

    def test_saved_doe_to_four_reconstructions_and_png_files(self):
        # Только тестовый ДОЭ: проверка сквозного процесса, без нейросетевого синтеза.
        doe_path = self.path.parent / "doe.png"
        doe_pixels = np.arange(128 * 128).reshape(128, 128).astype(np.uint8)
        Image.fromarray(doe_pixels).save(doe_path)
        with Image.open(doe_path) as image:
            phases = np.array(image, dtype=np.float64) / 255 * (2 * np.pi) - np.pi
        field = slm_field_matrix(phases)
        intensities = np.stack([
            reconstruct_intensity(field, distance) for distance in RECONSTRUCTION_DISTANCES_M
        ])

        OutputPictures(self.path, intensities).save()
        loaded = InputPictures(self.path)
        loaded.load()

        saved = loaded.to_numpy()
        self.assertEqual(saved.shape, (4, 128, 128))
        for index in range(4):
            expected = np.rint(255 * intensities[index] / intensities[index].max()).astype(np.uint8)
            np.testing.assert_array_equal(saved[index], expected)


if __name__ == "__main__":
    unittest.main()
