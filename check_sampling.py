"""Воспроизводимая диагностика дискретизации одно-FFT модели Френеля.

Запуск: python check_sampling.py [--output sampling_report.json]

Сравниваются два разных приближения одного интеграла Френеля:
1. Существующий FFT и прямая сумма по центрам пикселей.
2. Кусочно-постоянные квадратные пиксели с заполнением 100%, интегрирование
   ядра внутри каждого пикселя методом средних прямоугольников (64 и 128
   подшагов на ось). Пиксель занимает center +/- PIXEL_PITCH_M/2.

Вторая модель вводится ТОЛЬКО как независимая проверка чувствительности
к разрешению квадратуры. Она не заменяет рабочий одно-FFT алгоритм и не
доказывает точность физической установки или параксиального приближения.
Ни нормировка яркости, ни квантование фазы здесь не применяются.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from const import (
    IMAGE_SHAPE,
    IMAGE_SIZE,
    PIXEL_PITCH_M,
    RECONSTRUCTION_DISTANCES_M,
    WAVELENGTH_M,
    WAVE_NUMBER_PER_M,
)
from mathlib import fresnel_field_matrix


SAMPLE_PIXELS = ((64, 64), (65, 60), (15, 80), (0, 0), (127, 127))


def integrated_samples(field, distance_m, substeps):
    """Интеграл Френеля в пяти точках для постоянного поля внутри пикселя.

    field[row, col]: строка соответствует v/y, столбец — u/x.
    Разделимость ядра позволяет интегрировать по двум осям независимо;
    множители du внутри wx/wy учитывают полную площадь каждого пикселя.
    substeps=1 даёт ту же дискретную сумму, что и рабочий FFT.
    """
    axis = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * PIXEL_PITCH_M
    offsets = ((np.arange(substeps) + 0.5) / substeps - 0.5) * PIXEL_PITCH_M
    coordinates = axis[:, None] + offsets
    pitch = WAVELENGTH_M * distance_m / (IMAGE_SIZE * PIXEL_PITCH_M)
    prefactor = np.exp(1j * WAVE_NUMBER_PER_M * distance_m) / (
        1j * WAVELENGTH_M * distance_m
    )
    result = []
    for row, col in SAMPLE_PIXELS:
        x = (col - IMAGE_SIZE // 2) * pitch
        y = (row - IMAGE_SIZE // 2) * pitch
        wx = np.exp(
            1j * np.pi * (x - coordinates) ** 2 / (WAVELENGTH_M * distance_m)
        ).mean(axis=1) * PIXEL_PITCH_M
        wy = np.exp(
            1j * np.pi * (y - coordinates) ** 2 / (WAVELENGTH_M * distance_m)
        ).mean(axis=1) * PIXEL_PITCH_M
        result.append(prefactor * (wy @ field @ wx))
    return np.asarray(result, dtype=np.complex128)


def relative_l2(actual, reference):
    """Относительная L2-ошибка комплексных полей в выбранных пяти точках."""
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def sampling_report():
    """Собрать диагностический JSON без изменения рабочих параметров модели."""
    rng = np.random.default_rng(42)
    fields = {
        "zero_phase": np.ones(IMAGE_SHAPE, dtype=np.complex128),
        "uniform_random_phase_seed_42": np.exp(
            1j * rng.uniform(-np.pi, np.pi, IMAGE_SHAPE)
        ),
    }
    report = {
        "description": (
            "FFT agreement with a discrete sum and sensitivity to subpixel "
            "integration of piecewise-constant square pixels; not full physical validation."
        ),
        "wavelength_m": WAVELENGTH_M,
        "N": IMAGE_SIZE,
        "input_pitch_m": PIXEL_PITCH_M,
        "input_width_m": IMAGE_SIZE * PIXEL_PITCH_M,
        "critical_distance_m": IMAGE_SIZE * PIXEL_PITCH_M**2 / WAVELENGTH_M,
        "sample_pixels_row_col": SAMPLE_PIXELS,
        "pixel_convention": "center +/- input_pitch/2; 100% fill; H constant per pixel",
        "relative_error_definition": "norm(actual-reference)/norm(reference), complex field",
        "reference_substeps_per_axis": [64, 128],
        "normalization": "none",
        "source": "https://doi.org/10.1364/AO.48.006132",
        "planes": [],
    }
    for distance in RECONSTRUCTION_DISTANCES_M:
        pitch = WAVELENGTH_M * distance / (IMAGE_SIZE * PIXEL_PITCH_M)
        # Максимальное приращение квадратичной фазы между СОСЕДНИМИ
        # отсчётами на оси [-64,...,63]; это диагностический показатель,
        # а не достаточное условие отсутствия алиасинга произвольного H.
        input_phase_step = np.pi * (IMAGE_SIZE - 1) * PIXEL_PITCH_M**2 / (
            WAVELENGTH_M * distance
        )
        output_phase_step = np.pi * (IMAGE_SIZE - 1) * pitch**2 / (
            WAVELENGTH_M * distance
        )
        transverse_bound = np.sqrt(2) * IMAGE_SIZE / 2 * (PIXEL_PITCH_M + pitch)
        plane = {
            "distance_m": distance,
            "output_pitch_m": pitch,
            "output_width_m": IMAGE_SIZE * pitch,
            "input_chirp_max_adjacent_step_rad": float(input_phase_step),
            "output_chirp_max_adjacent_step_rad": float(output_phase_step),
            "output_chirp_adjacent_step_exceeds_pi": bool(output_phase_step > np.pi),
            "paraxial_neglected_quartic_phase_estimate_rad": float(
                WAVE_NUMBER_PER_M * transverse_bound**4 / (8 * distance**3)
            ),
            "cases": {},
        }
        for name, field in fields.items():
            fft_field = fresnel_field_matrix(field, distance)
            fft_samples = np.asarray([fft_field[row, col] for row, col in SAMPLE_PIXELS])
            direct_samples = integrated_samples(field, distance, 1)
            reference_64 = integrated_samples(field, distance, 64)
            reference_128 = integrated_samples(field, distance, 128)
            plane["cases"][name] = {
                "fft_vs_discrete_sum_relative_l2": relative_l2(fft_samples, direct_samples),
                "fft_vs_subpixel128_relative_l2": relative_l2(fft_samples, reference_128),
                "subpixel64_vs128_relative_l2": relative_l2(reference_64, reference_128),
                "fft_sample_intensities": (np.abs(fft_samples) ** 2).tolist(),
                "subpixel128_sample_intensities": (np.abs(reference_128) ** 2).tolist(),
            }
        report["planes"].append(plane)
    report["limitations"] = [
        "Five selected output points do not establish a whole-image error bound.",
        "64-to-128 convergence is an empirical check, not a rigorous error estimate.",
        "Discrete power conservation follows from Parseval and does not prove continuum accuracy.",
        "The output chirp cancels in intensity but affects the sampled complex phase.",
        "Random pixel phase is not band limited; input chirp sampling alone cannot exclude aliases.",
        "The reference still uses scalar paraxial Fresnel propagation; no vector or nonparaxial validation.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Также записать JSON в указанный файл.")
    arguments = parser.parse_args()
    serialized = json.dumps(sampling_report(), ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if arguments.output is not None:
        arguments.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
