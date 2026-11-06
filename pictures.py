from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from const import (
    IMAGE_FORMAT,
    IMAGE_MODE,
    IMAGE_SHAPE,
    INPUT_IMAGE_COUNT,
    MAX_PIXEL_VALUE,
)


class InputPictures:
    """Четыре входных изображения в порядке сортировки имён файлов.

    Порядок соответствует плоскостям 24,1 / 25,3 / 26,5 / 27,7 мм.
    Исходные пиксели хранятся как uint8 без изменения размера и каналов.
    """

    def __init__(self, path_input: str | Path = Path("input")) -> None:
        self.path_input = Path(path_input)
        self.images: list[NDArray[np.uint8]] = []

    def load(self) -> None:
        """Загрузить все файлы папки: ровно четыре grayscale PNG 128×128.

        TypeError: число файлов не равно четырём (учитываются все форматы).
        ValueError: файл не читается или не соответствует формату из ТЗ.
        Ошибка загрузки сохраняет ранее загруженные изображения.
        """
        files = sorted(
            (path for path in self.path_input.iterdir() if path.is_file()),
            key=lambda path: path.name,
        )
        if len(files) != INPUT_IMAGE_COUNT:
            raise TypeError(
                f"В папке {self.path_input} должно быть ровно {INPUT_IMAGE_COUNT} файла; "
                f"найдено: {len(files)}."
            )

        images = []
        for path in files:
            try:
                with Image.open(path) as image:
                    if image.format != IMAGE_FORMAT:
                        raise ValueError(f"Файл {path} должен быть в формате PNG.")
                    if image.size != IMAGE_SHAPE:
                        raise ValueError(f"Изображение {path} должно иметь размер 128×128.")
                    if image.mode != IMAGE_MODE:
                        raise ValueError(
                            f"Изображение {path} должно быть 8-битным одноканальным "
                            f"grayscale (режим L); получен режим {image.mode}."
                        )
                    images.append(np.array(image, dtype=np.uint8))
            except OSError as error:
                raise ValueError(f"Не удалось прочитать изображение {path}.") from error

        self.images = images

    def to_numpy(self) -> NDArray[np.uint8]:
        """Вернуть новый массив (4, 128, 128), uint8, диапазон [0, 255]."""
        if len(self.images) != INPUT_IMAGE_COUNT:
            raise ValueError("Сначала загрузите четыре изображения методом load().")
        return np.stack(self.images, axis=0)

    def normalize(self) -> NDArray[np.float32]:
        """Вернуть новый массив float32 в [0, 1] делением пикселей на 255.

        Форма (4, 128, 128) и исходные значения self.images сохраняются.
        """
        return self.to_numpy().astype(np.float32) / np.float32(MAX_PIXEL_VALUE)


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
        FileExistsError: папка занята файлом или одно из имён уже существует.
        Остальные ошибки записи передаются вызывающему коду как OSError.
        """
        pixels = self.to_uint8()
        paths = [self.path_output / f"{index}.png" for index in range(1, INPUT_IMAGE_COUNT + 1)]
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Файл уже существует: {path}")
        self.path_output.mkdir(parents=True, exist_ok=True)
        for image_pixels, path in zip(pixels, paths):
            with path.open("xb") as stream:
                Image.fromarray(image_pixels).save(stream, format=IMAGE_FORMAT)
        return paths
