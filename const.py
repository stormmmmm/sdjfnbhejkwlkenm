"""Константы ТЗ: длины в метрах, фаза в радианах."""

from math import pi as PI


# Формат входных изображений, фазового ДОЭ и реконструкций.
INPUT_IMAGE_COUNT = 4
IMAGE_SIZE = 128
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_FORMAT = "PNG"
IMAGE_MODE = "L"
IMAGE_BIT_DEPTH = 8
MAX_PIXEL_VALUE = 2**IMAGE_BIT_DEPTH - 1
PHASE_MIN_RAD = -PI
PHASE_MAX_RAD = PI

# Физические параметры ТЗ в СИ. Расстояния соответствуют изображениям 1–4.
WAVELENGTH_M = 532e-9
PIXEL_PITCH_M = 10e-6
RECONSTRUCTION_DISTANCES_M = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
WAVE_NUMBER_PER_M = 2 * PI / WAVELENGTH_M

# Допущение модели освещения, а не требование ТЗ.
DEFAULT_AMPLITUDE = 1.0
