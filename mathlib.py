"""Поле фазового ДОЭ и восстановление одно-FFT методом Френеля.

Все координаты и расстояния — в метрах, фаза — в радианах.
slm_field_matrix собирает поле H размером 128×128, fresnel_field_matrix
рассчитывает поле K, reconstruct_intensity возвращает интенсивность |K|².
Для каждой плоскости выполняется один общий 2D FFT всего входного поля.
Синтез фазы нейросетью реализуется отдельно.
"""

import cmath
import math

import numpy as np
from numpy.typing import NDArray

from const import (
    DEFAULT_AMPLITUDE,
    IMAGE_SHAPE,
    IMAGE_SIZE,
    PHASE_MAX_RAD,
    PHASE_MIN_RAD,
    PI,
    PIXEL_PITCH_M,
    WAVELENGTH_M,
    WAVE_NUMBER_PER_M,
)


def _positive_distance(distance_m: float) -> float:
    """Проверить конечность и положительность расстояния в метрах."""
    return math.isfinite(distance_m) or distance_m <= 0


def slm_field(phase: float, amplitude: float = DEFAULT_AMPLITUDE) -> complex:
    """Рассчитать H(u, v) = A(u, v) * exp(1j*φ(u, v)) для одного пикселя.

    phase — вещественная фаза этого пикселя в диапазоне [−π, π], рад.
    amplitude — конечная неотрицательная амплитуда падающего поля в нём.
    По умолчанию A = 1: допущение равномерного освещения единичной амплитуды.
    Интенсивность в этой точке равна abs(H)**2 = A**2.

    Для финальной реконструкции phase должна быть декодирована из
    сохранённого PNG ДОЭ: phase = pixel/255 * (2*π) - π.
    Возвращает одно число complex. Нормировка и квантование не выполняются.
    ValueError: неверный диапазон фазы, неконечная амплитуда или A < 0.
    """
    if not PHASE_MIN_RAD <= phase <= PHASE_MAX_RAD:
        raise ValueError("Фаза должна находиться в диапазоне [−π, π].")
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Амплитуда должна быть конечной и неотрицательной.")
    return amplitude * cmath.exp(1j * phase)


def slm_field_matrix(
    phases: NDArray[np.floating], amplitude: float = DEFAULT_AMPLITUDE
) -> NDArray[np.complex128]:
    """Собрать H[m, n] = amplitude * exp(1j*phases[m, n]) для всего ДОЭ.

    phases — вещественная матрица 128×128 в радианах, значения в [−π, π].
    amplitude — общая конечная неотрицательная амплитуда освещения;
    по умолчанию A = 1. Возвращается новая матрица complex128.
    Проверка значений выполняется существующей функцией slm_field.
    Для финальной реконструкции phases декодируют из сохранённого PNG ДОЭ.
    FFT, нормировка, квантование и изменение исходной матрицы не выполняются.
    ValueError: неверная форма, тип или значения фазовой карты / амплитуды.
    """
    phases = np.asarray(phases)
    if phases.shape != IMAGE_SHAPE:
        raise ValueError("Фазовая карта должна иметь размер 128×128.")
    if phases.dtype.kind not in "fiu":
        raise ValueError("Фазовая карта должна содержать вещественные числа в радианах.")
    return np.array(
        [[slm_field(float(phase), amplitude) for phase in row] for row in phases],
        dtype=np.complex128,
    )


def fresnel_input(field: complex, u_m: float, v_m: float, distance_m: float) -> complex:
    """Подготовить один входной пиксель для общего 2D FFT.

    Возвращает H(u, v) * exp(1j*π*(u²+v²)/(λ*z)).
    field — одно значение H; u_m, v_m — его координаты относительно
    центрального пикселя ПВМС; distance_m — положительная дистанция z.
    Все длины заданы в метрах. Само FFT здесь не выполняется.

    Вызывающий код собирает подготовленные значения всего входного поля
    и выполняет fftshift(fft2(ifftshift(prepared_field))) один раз на плоскость,
    со стандартной нормировкой NumPy (прямое FFT без масштабирования).
    ValueError: неположительная или неконечная дистанция.
    """
    assert _positive_distance(distance_m)
    return field * cmath.exp(1j * PI * (u_m**2 + v_m**2) / (WAVELENGTH_M * distance_m))


def fresnel_propagate(
    fft_value: complex, x_m: float, y_m: float, distance_m: float
) -> complex:
    """Рассчитать K(x, y) для одного выходного пикселя по коэффициенту FFT.

    fft_value — один коэффициент центрированного 2D FFT всего поля,
    подготовленного через fresnel_input. Это не значение H одного пикселя.
    x_m, y_m — координаты выходного пикселя; distance_m — дистанция z > 0.
    Все длины — в метрах. Для массива N×N шаг выходной сетки равен
    λ*z/(N*PIXEL_PITCH_M), координата — (index - N//2) * шаг.

    K = exp(1j*k*z)/(1j*λ*z) * exp(1j*π*(x²+y²)/(λ*z))
        * fft_value * PIXEL_PITCH_M².
    Площадь входного пикселя учитывает дискретизацию интеграла Френеля.
    Возвращает одно число complex; интенсивность I = abs(K)**2.
    Нормировка яркости не выполняется. Функция сама FFT не вызывает.
    ValueError: неположительная или неконечная дистанция.
    """
    assert _positive_distance(distance_m)
    prefactor = cmath.exp(1j * WAVE_NUMBER_PER_M * distance_m) / (1j * WAVELENGTH_M * distance_m)
    output_chirp = cmath.exp(1j * PI * (x_m**2 + y_m**2) / (WAVELENGTH_M * distance_m))
    return prefactor * output_chirp * fft_value * PIXEL_PITCH_M**2


def _prepare_fresnel_field(
    field: NDArray[np.complexfloating], distance_m: float
) -> NDArray[np.complex128]:
    """Проверить входное поле H и собрать матрицу входных множителей Френеля."""
    field = np.asarray(field)
    if field.shape != IMAGE_SHAPE:
        raise ValueError("Поле H должно иметь размер 128×128.")
    if field.dtype.kind not in "fciu" or not np.isfinite(field).all():
        raise ValueError("Поле H должно содержать конечные вещественные или комплексные числа.")
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Расстояние должно быть положительным и конечным, в метрах.")
    axis_m = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * PIXEL_PITCH_M
    return np.array(
        [[fresnel_input(field[row, col], u_m, v_m, distance_m)
          for col, u_m in enumerate(axis_m)]
         for row, v_m in enumerate(axis_m)],
        dtype=np.complex128,
    )


def fresnel_field_matrix(
    field: NDArray[np.complexfloating], distance_m: float
) -> NDArray[np.complex128]:
    """Рассчитать комплексное поле K размером 128×128 из входного поля H.

    distance_m — положительное конечное расстояние z в метрах.
    Используется один центрированный 2D FFT с прямой нормировкой NumPy.
    Выходной шаг: λ*z/(128*PIXEL_PITCH_M); координаты центра равны нулю.
    Для K[row, col]: x = (col - 64)*шаг, y = (row - 64)*шаг.
    Масштаб 1/(i*λ*z), площадь входного пикселя и выходная фаза
    учитываются в fresnel_propagate. Входная матрица не изменяется.
    ValueError: неверная форма, неконечные / нечисловые данные или z <= 0.
    """
    prepared = _prepare_fresnel_field(field, distance_m)
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(prepared)))
    output_pitch_m = WAVELENGTH_M * distance_m / (IMAGE_SIZE * PIXEL_PITCH_M)
    axis_m = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * output_pitch_m
    return np.array(
        [[fresnel_propagate(spectrum[row, col], x_m, y_m, distance_m)
          for col, x_m in enumerate(axis_m)]
         for row, y_m in enumerate(axis_m)],
        dtype=np.complex128,
    )


def reconstruct_intensity(
    field: NDArray[np.complexfloating], distance_m: float
) -> NDArray[np.float64]:
    """Восстановить изображение I_z = |K_z|² из поля H на расстоянии z.

    Возвращает новую матрицу float64 размером 128×128 без нормировки
    яркости и преобразования в uint8. Физическая сетка зависит от z;
    её шаг указан в fresnel_field_matrix. Для четырёх плоскостей
    функцию вызывают с одним и тем же H и разными расстояниями.
    Для финальных результатов H собирают из фазы сохранённого PNG ДОЭ.
    """
    output_field = fresnel_field_matrix(field, distance_m)
    return np.abs(output_field)**2
