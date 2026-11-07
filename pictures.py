from io import StringIO
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
    PHASE_MIN_RAD,
    PHASE_MAX_RAD,
)


def phase_to_uint8(phase: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Закодировать [-π, π] фиксированной шкалой, без нормировки по данным."""
    phase = np.asarray(phase)
    if phase.shape != IMAGE_SHAPE or phase.dtype.kind not in "fiu":
        raise ValueError("Фаза должна быть вещественной матрицей 128×128.")
    if (not np.isfinite(phase).all() or (phase < PHASE_MIN_RAD).any()
            or (phase > PHASE_MAX_RAD).any()):
        raise ValueError("Все значения фазы должны быть конечными и лежать в [−π, π].")
    pixels = np.rint(MAX_PIXEL_VALUE * (phase - PHASE_MIN_RAD)
                     / (PHASE_MAX_RAD - PHASE_MIN_RAD))
    return np.clip(pixels, 0, MAX_PIXEL_VALUE).astype(np.uint8)


def load_phase_png(path: str | Path) -> NDArray[np.float64]:
    """Прочитать только однокадровый PNG L 128×128 и декодировать радианы.

    RGB, палитра, 16 бит и изменение размеров автоматически не допускаются.
    """
    try:
        with Image.open(path) as image:
            if image.format != IMAGE_FORMAT:
                raise ValueError("Фазовая маска должна быть в формате PNG.")
            if image.size != IMAGE_SHAPE:
                raise ValueError("Фазовая маска должна иметь размер 128×128.")
            if image.mode != IMAGE_MODE:
                raise ValueError("Нужен одноканальный 8-битный PNG (режим L), без RGB и палитры.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Нужен статический PNG, а не анимация.")
            pixels = np.array(image, dtype=np.float64)
    except OSError as error:
        raise ValueError(f"Не удалось прочитать фазовую маску: {path}") from error
    return pixels / MAX_PIXEL_VALUE * (PHASE_MAX_RAD - PHASE_MIN_RAD) + PHASE_MIN_RAD


def load_phase_txt(path: str | Path) -> NDArray[np.float64]:
    """Прочитать TXT: 128 строк по 128 фаз в радианах [-π, π].

    UTF-8 (включая BOM); разделители: пробел, табуляция, запятая, точка
    с запятой. Десятичный знак — точка, экспоненциальная запись допустима.
    Пустые строки и комментарии после # игнорируются. Никаких нормировки,
    квантования, изменения формы или циклической обёртки фаз не выполняется.
    """
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Не удалось прочитать TXT с фазами: {path}") from error
    rows = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    text = "\n".join(line for line in rows if line)
    if not text:
        raise ValueError("TXT-файл пуст: нужны 128 строк по 128 фаз в радианах.")
    try:
        phase = np.loadtxt(StringIO(text.replace(",", " ").replace(";", " ")),
                           dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError("Некорректная TXT-матрица: нужны строки чисел одинаковой длины; "
                         "десятичный знак — точка.") from error
    # TXT-экспорт с 9 знаками пишет pi как 3.141592654, чуть выше pi.
    # Исправляем только превышение границы <= 1e-9 рад, не оборачивая фазы.
    tolerance_rad = 1e-9
    near_upper = ((phase > PHASE_MAX_RAD)
                  & (phase <= PHASE_MAX_RAD + tolerance_rad))
    near_lower = ((phase < PHASE_MIN_RAD)
                  & (phase >= PHASE_MIN_RAD - tolerance_rad))
    phase[near_upper] = PHASE_MAX_RAD
    phase[near_lower] = PHASE_MIN_RAD
    # Общая строгая проверка после устранения погрешности записи границ.
    # Кодировка используется только для проверки, квантования результата нет.
    phase_to_uint8(phase)
    return phase


def save_phase_png(phase: NDArray[np.floating], path: str | Path) -> Path:
    """Сохранить фазу как PNG L 128×128; существующий файл не перезаписывать."""
    pixels = phase_to_uint8(phase)
    path = Path(path)
    with path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format=IMAGE_FORMAT)
    return path


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
