"""Сгенерировать синтетические четвёрки Френеля в dataset/{train,test,validation}.

Допущение генератора: пиксели каждой фазы независимы и равномерны в [−π, π),
освещение равномерное A=1. Это синтетические данные, не изображения кейса.
Фазы непрерывные float64, без PNG-квантования; они воспроизводятся по seed.
Финальные ДОЭ сети по-прежнему оцениваются после сохранения и загрузки PNG.
"""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.const import (
    DATASET_DIRECTORY,
    IMAGE_SHAPE,
    INPUT_IMAGE_COUNT,
    PHASE_MAX_RAD,
    PHASE_MIN_RAD,
)
from utils.mathlib import FresnelPropagation
from utils.pictures import OutputPictures

SEED = 42
SPLIT_NAMES = ("train", "test", "validation")
SPLIT_SIZES = (800, 100, 100)
PROGRESS_INTERVAL = 100


def generate_dataset(
    output_directory: str | Path = DATASET_DIRECTORY,
    seed: int = SEED,
    split_sizes: tuple[int, int, int] = SPLIT_SIZES,
) -> None:
    """Сохранить по четыре PNG L/uint8/128×128 для каждой случайной фазы.

    split_sizes задаёт число фаз в train/test/validation. Один локальный
    генератор NumPy PCG64 с seed последовательно создаёт фазы 128×128, рад.
    Четвёрка целиком принадлежит одному сплиту. Нумерация в каждом — с 1.
    Расчёт на CPU использует float64/complex128 и физику mathlib: один FFT
    на плоскость, z=24,1/25,3/26,5/27,7 мм, λ=532 нм, шаг ПВМС 10 мкм.
    Выходной шаг λ*z/(128*10 мкм) различается между плоскостями.
    OutputPictures нормирует каждую интенсивность как round(255*I/max(I)).

    Существование любого из трёх каталогов сплитов вызывает FileExistsError
    до записи файлов: повторный запуск не смешивает и не затирает наборы.
    Фазы и веса сети не сохраняются; обучение здесь не выполняется.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed должен быть неотрицательным целым числом.")
    if (
        len(split_sizes) != len(SPLIT_NAMES)
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in split_sizes
        )
        or sum(split_sizes) == 0
    ):
        raise ValueError(
            "Нужны три неотрицательных целых размера train/test/validation, сумма > 0."
        )

    output_directory = Path(output_directory)
    split_directories = [output_directory / name for name in SPLIT_NAMES]
    for directory in split_directories:
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(
                f"Каталог сплита уже существует: {directory}. "
                "Для нового набора выберите другую выходную папку."
            )

    rng = np.random.Generator(np.random.PCG64(seed))
    total_samples = sum(split_sizes)
    logging.info(
        "Синтетический набор: seed=%d, сплиты=%s, путь=%s", seed, split_sizes, output_directory
    )
    with tf.device("/CPU:0"):
        propagation = FresnelPropagation(dtype=tf.float64)
        completed = 0
        for split_directory, sample_count in zip(split_directories, split_sizes):
            split_directory.mkdir(parents=True, exist_ok=False)
            for sample_index in range(1, sample_count + 1):
                phase_map = rng.uniform(PHASE_MIN_RAD, PHASE_MAX_RAD, size=IMAGE_SHAPE)
                intensities = propagation(tf.constant(phase_map[None], dtype=tf.float64))[0].numpy()
                OutputPictures(split_directory / str(sample_index), intensities).save()
                completed += 1
                if completed % PROGRESS_INTERVAL == 0 or completed == total_samples:
                    logging.info("Сгенерировано фаз: %d/%d", completed, total_samples)
            logging.info("%s: %d четвёрок", split_directory.name, sample_count)
    logging.info(
        "Готово: %d фаз, %d PNG.",
        total_samples,
        total_samples * INPUT_IMAGE_COUNT,
    )


def main() -> None:
    """Создать 1000 фаз и 4000 изображений с настройками, заданными в коде."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_dataset()


if __name__ == "__main__":
    main()
