"""Физическое восстановление и оценка четвёрки по сохранённому ДОЭ."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from model.metrics import calculate_pair_metrics
from utils import const
from utils.mathlib import FresnelPropagation
from utils.pictures import InputPictures, OutputPictures, load_phase


class PlaneMetrics(TypedDict):
    """Метрики одной плоскости в порядке расстояний из ТЗ."""

    plane: int
    distance_m: float
    pearson: float | None
    ssim: float


@dataclass(frozen=True)
class QuartetEvaluation:
    """Декодированная фаза H×W в радианах и реконструкции 4×H×W в [0,255]."""

    phase: NDArray[np.float64]
    reconstructions: NDArray[np.uint8]
    planes: list[PlaneMetrics]


def _peak_normalize(images: NDArray[np.float64]) -> NDArray[np.float64]:
    """Нормировать каждую плоскость 4×H×W по максимуму, сохранив нулевые плоскости."""
    peaks = images.max(axis=(-2, -1), keepdims=True)
    return np.divide(images, peaks, out=np.zeros_like(images), where=peaks > 0)


class SavedDoeEvaluator:
    """Повторно читать фазовый PNG и рассчитывать поле Френеля в float64."""

    def __init__(self) -> None:
        self._propagation = FresnelPropagation(dtype=tf.float64)

    @property
    def output_pixel_pitch_m(self) -> list[float]:
        """Шаг выходной сетки каждой плоскости в метрах."""
        return self._propagation.output_pixel_pitch_m.numpy().tolist()

    def evaluate(
        self,
        doe_path: Path,
        reconstruction_directory: Path,
        targets: NDArray[np.uint8],
        *,
        compare_amplitudes: bool = False,
    ) -> QuartetEvaluation:
        """Сохранить интенсивности и сравнить их PNG с целями 4×H×W.

        Для экспертной оценки compare_amplitudes=True сравнивает цели
        и амплитуды поля до округления реконструкций, с максимумом 1.
        """
        if (
            targets.shape != (const.INPUT_IMAGE_COUNT, *const.IMAGE_SHAPE)
            or targets.dtype != np.uint8
        ):
            raise ValueError("Для оценки нужны четыре цели 128×128 в формате uint8.")
        phase = load_phase(doe_path)
        intensities = self._propagation(tf.constant(phase[None], tf.float64))[0].numpy()
        OutputPictures(reconstruction_directory, intensities).save()
        reconstructions = InputPictures(reconstruction_directory)
        reconstructions.load()
        pixels = reconstructions.to_numpy()
        if compare_amplitudes:
            reference = _peak_normalize(targets.astype(np.float64))
            comparison = _peak_normalize(np.sqrt(intensities))
            data_range = 1.0
        else:
            reference = targets
            comparison = pixels
            data_range = const.MAX_PIXEL_VALUE
        planes: list[PlaneMetrics] = []
        for index, (target, reconstructed, distance_m) in enumerate(
            zip(reference, comparison, const.RECONSTRUCTION_DISTANCES_M, strict=True), start=1
        ):
            pearson, ssim = calculate_pair_metrics(target, reconstructed, data_range=data_range)
            planes.append(
                {
                    "plane": index,
                    "distance_m": distance_m,
                    "pearson": pearson if np.isfinite(pearson) else None,
                    "ssim": ssim,
                }
            )
        return QuartetEvaluation(phase, pixels, planes)
