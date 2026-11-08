from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from utils.const import (
    IMAGE_FORMAT,
    IMAGE_MODE,
    IMAGE_SHAPE,
    INPUT_IMAGE_COUNT,
    MAX_PIXEL_VALUE,
)


def load_grayscale_png(path: Path, *, contents: bytes | None = None) -> NDArray[np.uint8]:
    """Декодировать PNG из файла или уже прочитанных байтов без изменения пикселей."""
    if contents is None and not path.is_file():
        raise FileNotFoundError(f"Не найдено изображение: {path}")
    try:
        with Image.open(path if contents is None else BytesIO(contents)) as image:
            if (
                image.format != IMAGE_FORMAT
                or image.size != IMAGE_SHAPE
                or image.mode != IMAGE_MODE
            ):
                raise ValueError(f"Изображение {path} должно быть grayscale PNG (L), 128×128.")
            return np.array(image, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"Не удалось прочитать изображение {path}.") from error


def normalize_pixels(pixels: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Перевести пиксели uint8 в float32 [0, 1], сохраняя форму."""
    return pixels.astype(np.float32) / np.float32(MAX_PIXEL_VALUE)


def save_phase(path: str | Path, phase_map: NDArray[np.floating]) -> None:
    """Сохранить вещественную фазу 128×128: −π → 0, +π → 255.

    Фаза задаётся в радианах; округлённая кодировка ограничивается [0,255].
    Нечисловые, комплексные, логические и неконечные данные отвергаются.
    """
    phase_map = np.asarray(phase_map)
    if phase_map.dtype.kind not in "fiu":
        raise ValueError("Фаза должна содержать вещественные числа в радианах.")
    phase_map = phase_map.astype(np.float64)
    if phase_map.shape != IMAGE_SHAPE or not np.isfinite(phase_map).all():
        raise ValueError("Фаза должна быть конечным массивом формы (128, 128).")
    pixels = np.clip(
        np.rint(MAX_PIXEL_VALUE * (phase_map + np.pi) / (2 * np.pi)),
        0,
        MAX_PIXEL_VALUE,
    ).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format=IMAGE_FORMAT)


def load_phase(path: str | Path) -> NDArray[np.float64]:
    """Декодировать сохранённый 8-битный ДОЭ 128×128 в фазу [−π, π]."""
    pixels = load_grayscale_png(Path(path))
    return pixels.astype(np.float64) / MAX_PIXEL_VALUE * (2 * np.pi) - np.pi


class InputPictures:
    """Четыре входных изображения с именами 1.png, 2.png, 3.png, 4.png.

    Порядок соответствует плоскостям 24,1 / 25,3 / 26,5 / 27,7 мм.
    Исходные пиксели хранятся как uint8 без изменения размера и каналов.
    """

    def __init__(self, path_input: str | Path = Path("../input")) -> None:
        self.path_input = Path(path_input)
        self.images: list[NDArray[np.uint8]] = []

    def load(self) -> None:
        """Загрузить 1.png–4.png: grayscale PNG 128×128.

        Дополнительные файлы и папки игнорируются.
        ValueError: файл не читается или не соответствует формату из ТЗ.
        Ошибка загрузки сохраняет ранее загруженные изображения.
        """
        self.images = [
            load_grayscale_png(self.path_input / f"{index}.png")
            for index in range(1, INPUT_IMAGE_COUNT + 1)
        ]

    def to_numpy(self) -> NDArray[np.uint8]:
        """Вернуть новый массив (4, 128, 128), uint8, диапазон [0, 255]."""
        if len(self.images) != INPUT_IMAGE_COUNT:
            raise ValueError("Сначала загрузите четыре изображения методом load().")
        return np.stack(self.images, axis=0)

    def normalize(self) -> NDArray[np.float32]:
        """Вернуть новый массив float32 в [0, 1] делением пикселей на 255.

        Форма (4, 128, 128) и исходные значения self.images сохраняются.
        """
        return normalize_pixels(self.to_numpy())


class OutputPictures:
    """Четыре матрицы интенсивности в порядке плоскостей восстановления.

    Порядок: 24,1 / 25,3 / 26,5 / 27,7 мм. Исходные интенсивности
    копируются, чтобы обработка и сохранение не изменяли входной массив.
    """

    def __init__(self, path_output: str | Path, intensities: NDArray[np.floating]) -> None:
        """Принять конечные неотрицательные интенсивности формы (4, 128, 128).

        ValueError: неверная форма, нечисловой / комплексный тип,
        отрицательные или неконечные значения интенсивности.
        """
        intensities = np.asarray(intensities)
        if intensities.shape != (INPUT_IMAGE_COUNT, *IMAGE_SHAPE):
            raise ValueError("Интенсивности должны иметь форму (4, 128, 128).")
        if intensities.dtype.kind not in "fiu":
            raise ValueError("Интенсивности должны содержать вещественные числа.")
        if not np.isfinite(intensities).all() or (intensities < 0).any():
            raise ValueError("Интенсивности должны быть конечными и неотрицательными.")
        self.path_output = Path(path_output)
        self.intensities = intensities.astype(np.float64, copy=True)

    def to_uint8(self) -> NDArray[np.uint8]:
        """Вернуть четыре изображения uint8, нормированные каждое по своему максимуму.

        pixels = round(255 * I / max(I)), с ограничением в [0, 255].
        Нулевая матрица даёт чёрное изображение. Вычитание минимума,
        гамма-коррекция и изменение исходных интенсивностей не выполняются.
        Эта нормировка убирает различия абсолютной яркости между плоскостями.
        """
        peaks = self.intensities.max(axis=(1, 2), keepdims=True)
        normalized = np.divide(
            self.intensities, peaks, out=np.zeros_like(self.intensities), where=peaks > 0
        )
        return np.clip(np.rint(MAX_PIXEL_VALUE * normalized), 0, MAX_PIXEL_VALUE).astype(np.uint8)

    def save(self) -> list[Path]:
        """Обработать интенсивности и сохранить четыре grayscale PNG 128×128.

        Папка создаётся вместе с родительскими каталогами при необходимости.
        Имена 1.png–4.png соответствуют 24,1 / 25,3 / 26,5 / 27,7 мм.
        Возвращает пути файлов в том же порядке.
        Существующие изображения с теми же именами перезаписываются.
        """
        pixels = self.to_uint8()
        paths = [self.path_output / f"{index}.png" for index in range(1, INPUT_IMAGE_COUNT + 1)]
        self.path_output.mkdir(parents=True, exist_ok=True)
        for image_pixels, path in zip(pixels, paths):
            Image.fromarray(image_pixels).save(path, format=IMAGE_FORMAT)
        return paths
