"""Оркестрация симуляции и экспорт; формула распространения — только в mathlib.

GUI не участвует в физическом расчёте. При переходе на PyTorch можно заменить
mathlib и адаптеры этого модуля, сохранив границу simulate -> SimulationResult.
Текущий NumPy backend не дифференцируемый.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from const import (
    DEFAULT_AMPLITUDE, IMAGE_SHAPE, IMAGE_SIZE, PHASE_MIN_RAD, PHASE_MAX_RAD,
    PIXEL_PITCH_M, RECONSTRUCTION_DISTANCES_M, WAVELENGTH_M,
)
from mathlib import fresnel_field_matrix, output_pixel_pitch, slm_field_matrix
from pictures import OutputPictures, load_phase_png, load_phase_txt, save_phase_png


@dataclass(frozen=True)
class SimulationResult:
    """Снимок одного расчёта: H создан один раз, K и I — для четырёх z.

    phase[row, col]: u=(col-64)*Δu, v=(row-64)*Δu; meshgrid соответствует
    indexing='xy'. В выходных полях оси аналогичны, но шаг зависит от z.
    Все длины в СИ, интенсивности относительные при A=1 (не Вт/м²).
    """

    phase: NDArray[np.float64]
    field: NDArray[np.complex128]
    output_fields: tuple[NDArray[np.complex128], ...]
    intensities: NDArray[np.float64]
    seed: int | None
    source: str


def generate_random_phase(seed: int | None = None) -> NDArray[np.float64]:
    """Uniform[-π, π), float64; одинаковый неотрицательный seed воспроизводим."""
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Seed должен быть целым неотрицательным числом или None.")
    return np.random.default_rng(seed).uniform(PHASE_MIN_RAD, PHASE_MAX_RAD, IMAGE_SHAPE)


def phase_to_field(phase: NDArray[np.floating]) -> NDArray[np.complex128]:
    """H = exp(i*phase), равномерное освещение единичной амплитуды."""
    return slm_field_matrix(phase)


def simulate(
    phase: NDArray[np.floating], seed: int | None = None, source: str = "random"
) -> SimulationResult:
    """Из одной фазы создать одно H и четыре K; вернуть I без нормировки."""
    field = phase_to_field(phase)
    output_fields = tuple(fresnel_field_matrix(field, z) for z in RECONSTRUCTION_DISTANCES_M)
    intensities = np.stack([np.abs(output)**2 for output in output_fields])
    # Собственные массивы сохраняют результат при изменении исходной фазы.
    phase_copy = np.asarray(phase, dtype=np.float64).copy()
    for array in (phase_copy, field, *output_fields, intensities):
        array.setflags(write=False)
    return SimulationResult(phase_copy, field, output_fields, intensities,
                            int(seed) if seed is not None else None, source)


def simulate_phase_txt(path: str | Path) -> SimulationResult:
    """Загрузить TXT-фазу и рассчитать четыре поля K и интенсивности |K|².

    Возвращает тот же результат, что simulate: одна общая маска, A=1,
    четыре фиксированных расстояния. Интенсивности не нормируются.
    """
    return simulate(load_phase_txt(path), source=str(Path(path).resolve()))


def reconstruct_all(phase: NDArray[np.floating]) -> NDArray[np.float64]:
    """Четыре интенсивности формы (4,128,128) из одной фазы, без нормировки."""
    return simulate(phase).intensities


def _save_intensities(result: SimulationResult, directory: Path) -> None:
    OutputPictures(directory, result.intensities).save()
    for index, intensity in enumerate(result.intensities, start=1):
        with (directory / f"{index}.npy").open("xb") as stream:
            np.save(stream, intensity, allow_pickle=False)


def export_results(result: SimulationResult, directory: str | Path) -> Path:
    """Создать новую папку с точным результатом и отдельной PNG-реконструкцией.

    1.npy..4.npy — исходные float64 интенсивности, точно как в GUI.
    phase_radians.npy — полная точность исходной фазы. doe.png квантован в 8 бит.
    quantized/ — дополнительные реконструкции именно из сохранённого doe.png.
    Существующая папка запрещена, чтобы сохранять пользовательские данные.
    При ошибке записи частичная новая папка сохраняется для диагностики.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_phase_png(result.phase, directory / "doe.png")
    with (directory / "phase_radians.npy").open("xb") as stream:
        np.save(stream, result.phase, allow_pickle=False)
    _save_intensities(result, directory)

    saved_phase = load_phase_png(directory / "doe.png")
    quantized = simulate(saved_phase, seed=result.seed, source="saved doe.png")
    _save_intensities(quantized, directory / "quantized")
    pitches = [output_pixel_pitch(z) for z in RECONSTRUCTION_DISTANCES_M]
    metadata = {
        "schema_version": 1,
        "wavelength_m": WAVELENGTH_M,
        "n": IMAGE_SIZE,
        "pixel_pitch_m": PIXEL_PITCH_M,
        "doe_size_m": IMAGE_SIZE * PIXEL_PITCH_M,
        "distances_m": RECONSTRUCTION_DISTANCES_M,
        "output_pixel_pitches_m": pitches,
        "screen_sizes_m": [IMAGE_SIZE * pitch for pitch in pitches],
        "amplitude": DEFAULT_AMPLITUDE,
        "seed": result.seed,
        "source": result.source,
        "axes": "array[row,col]: x=(col-N/2)*pitch, y=(row-N/2)*pitch; meshgrid indexing=xy",
        "method": "Fresnel: fftshift(fft2(ifftshift(H*input_chirp))); one 2D FFT per plane",
        "normalization": "PNG: round(255*I/max(I)) independently per plane; zero stays zero; NPY: none",
        "intensity_units": "relative, A=1; not calibrated W/m^2",
        "raw_max_intensities": result.intensities.max(axis=(1, 2)).tolist(),
        "input_power_relative_m2": float(np.sum(np.abs(result.field)**2) * PIXEL_PITCH_M**2),
        "output_powers_relative_m2": [float(np.sum(i) * p**2)
                                      for i, p in zip(result.intensities, pitches)],
        "phase_encoding": "uint8=round(255*(phase+pi)/(2*pi)); 0=-pi, 255=+pi",
        "full_precision": "phase_radians.npy and 1.npy..4.npy exactly preserve GUI calculation",
        "quantized": {
            "directory": "quantized",
            "source": "doe.png reloaded and decoded before recalculation",
            "max_phase_error_rad": float(np.max(np.abs(result.phase - saved_phase))),
            "raw_max_intensities": quantized.intensities.max(axis=(1, 2)).tolist(),
        },
        "limitations": [
            "Scalar paraxial Fresnel approximation; ideal thin phase mask, uniform plane-wave illumination.",
            "Center-sampled quadrature, no integration inside pixels, fill factor or SLM calibration.",
            "DFT power conservation validates discrete scaling, not continuum optical accuracy.",
            "Near the input-chirp sampling limit at 24.1 mm; arbitrary random masks may alias.",
        ],
    }
    with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return directory
