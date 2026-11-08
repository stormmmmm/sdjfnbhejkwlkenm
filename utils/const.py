"""Пути проекта и константы ТЗ в метрах и радианах."""

from math import pi as PI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "model.keras"

# Формат входных изображений, фазового ДОЭ и реконструкций.
INPUT_IMAGE_COUNT = 4
IMAGE_SIZE = 128
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_FORMAT = "PNG"
IMAGE_MODE = "L"
IMAGE_BIT_DEPTH = 8
MAX_PIXEL_VALUE = 2**IMAGE_BIT_DEPTH - 1

# Константы SSIM для шкалы яркости [0, 255]: 6.5025 и 58.5225.
SSIM_C1 = (0.01 * MAX_PIXEL_VALUE) ** 2
SSIM_C2 = (0.03 * MAX_PIXEL_VALUE) ** 2

PHASE_MIN_RAD = -PI
PHASE_MAX_RAD = PI

# Физические параметры ТЗ в СИ. Расстояния соответствуют изображениям 1–4.
WAVELENGTH_M = 532e-9
PIXEL_PITCH_M = 10e-6
RECONSTRUCTION_DISTANCES_M = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
WAVE_NUMBER_PER_M = 2 * PI / WAVELENGTH_M

# Допущение модели освещения, а не требование ТЗ.
DEFAULT_AMPLITUDE = 1.0
