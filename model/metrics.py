"""Пирсон и выбранный глобальный SSIM для пары изображений 128×128.

Для четырёх плоскостей функции вызывают отдельно для каждой пары.
Изображения передают в общей шкале яркости [0, data_range].
Метрики не нормируют яркость и не изменяют входы.
Глобальный вариант SSIM — решение реализации; ТЗ не задаёт размер окна.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from utils import const


def prepare_images(
    first_image: NDArray[np.number],
    second_image: NDArray[np.number],
    *,
    data_range: float = const.MAX_PIXEL_VALUE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Проверить форму и пиксели в [0, data_range], перейти к float64."""
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Верхняя граница шкалы яркости должна быть конечной и положительной.")
    images = []
    for image in (first_image, second_image):
        image = np.asarray(image)
        if image.shape != const.IMAGE_SHAPE:
            raise ValueError("Каждое изображение должно иметь размер 128×128.")
        if image.dtype.kind not in "fiu" or not np.isfinite(image).all():
            raise ValueError("Изображения должны содержать конечные вещественные числа.")
        if np.any((image < 0) | (image > data_range)):
            raise ValueError(
                f"Значения пикселей изображений должны находиться в [0,{data_range:g}]."
            )
        images.append(image.astype(np.float64))
    return images[0], images[1]


@dataclass(frozen=True)
class _ImageStatistics:
    mean: np.float64
    centered: NDArray[np.float64]
    sum_squares: np.float64


def _image_statistics(image: NDArray[np.float64]) -> _ImageStatistics:
    mean = image.mean()
    centered = image - mean
    return _ImageStatistics(mean, centered, np.sum(centered**2))


def _has_constant_image(first: NDArray[np.float64], second: NDArray[np.float64]) -> bool:
    # Среднее постоянного float-массива может отличаться от пикселя из-за округления.
    return bool(np.all(first == first.flat[0]) or np.all(second == second.flat[0]))


def _pearson(first: _ImageStatistics, second: _ImageStatistics) -> float:
    if first.sum_squares == 0 or second.sum_squares == 0:
        return float("nan")
    correlation = np.sum(
        (first.centered / np.sqrt(first.sum_squares))
        * (second.centered / np.sqrt(second.sum_squares))
    )
    return float(np.clip(correlation, -1.0, 1.0))


def _ssim(
    first: _ImageStatistics, second: _ImageStatistics, data_range: float = const.MAX_PIXEL_VALUE
) -> float:
    covariance = np.mean(first.centered * second.centered)
    first_variance = first.sum_squares / first.centered.size
    second_variance = second.sum_squares / second.centered.size
    scale_squared = (data_range / const.MAX_PIXEL_VALUE) ** 2
    c1 = const.SSIM_C1 * scale_squared
    c2 = const.SSIM_C2 * scale_squared
    numerator = (2 * first.mean * second.mean + c1) * (2 * covariance + c2)
    denominator = (first.mean**2 + second.mean**2 + c1) * (first_variance + second_variance + c2)
    return float(numerator / denominator)


def calculate_pierce_corr(
    first_image: NDArray[np.number], second_image: NDArray[np.number]
) -> float:
    """Рассчитать коэффициент Пирсона по всем пикселям двух изображений.

    Возвращает float в [−1, 1]. Если хотя бы одно изображение постоянно,
    возвращает NaN: при нулевой дисперсии корреляция не определена,
    в том числе для двух одинаковых постоянных изображений.
    ValueError: неверная форма, тип или значения пикселей вне [0,255].
    """
    first, second = prepare_images(first_image, second_image)
    if _has_constant_image(first, second):
        return float("nan")
    return _pearson(_image_statistics(first), _image_statistics(second))


def calculate_ssim(first_image: NDArray[np.number], second_image: NDArray[np.number]) -> float:
    """Рассчитать выбранный глобальный SSIM без скользящих окон.

    Средние, дисперсии и ковариация вычисляются по всему изображению.
    Дисперсии и ковариация делятся на число пикселей (ddof=0).
    Используются C1 и C2 для шкалы [0, 255].
    Возвращает float; для одинаковых изображений, включая постоянные, 1.
    ValueError: неверная форма, тип или значения пикселей вне [0,255].
    """
    first, second = prepare_images(first_image, second_image)
    return _ssim(_image_statistics(first), _image_statistics(second))


def calculate_pair_metrics(
    first_image: NDArray[np.number],
    second_image: NDArray[np.number],
    *,
    data_range: float = const.MAX_PIXEL_VALUE,
) -> tuple[float, float]:
    """Рассчитать Pearson и глобальный SSIM по общей шкале [0, data_range]."""
    first, second = prepare_images(first_image, second_image, data_range=data_range)
    first_stats, second_stats = _image_statistics(first), _image_statistics(second)
    pearson = (
        float("nan") if _has_constant_image(first, second) else _pearson(first_stats, second_stats)
    )
    return pearson, _ssim(first_stats, second_stats, data_range)
