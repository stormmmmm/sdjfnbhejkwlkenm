"""Единый расчёт фазового ДОЭ и одно-FFT распространения Френеля.

TensorFlow используется и в обучении, и при восстановлении сохранённого ДОЭ.
Фаза задаётся в радианах, координаты и расстояния — в метрах. Равномерное
освещение A = DEFAULT_AMPLITUDE — допущение модели. Постоянные множители
Френеля вычисляются NumPy в float64 один раз при создании пропагатора;
обучаемые тензоры проходят только через TensorFlow, без перехода в NumPy.
Скалярный и матричный NumPy API сохранён как адаптеры общего расчёта.
"""

import math

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from utils.const import (
    DEFAULT_AMPLITUDE,
    IMAGE_SHAPE,
    IMAGE_SIZE,
    PHASE_MAX_RAD,
    PHASE_MIN_RAD,
    PI,
    PIXEL_PITCH_M,
    RECONSTRUCTION_DISTANCES_M,
    WAVE_NUMBER_PER_M,
    WAVELENGTH_M,
)


def _positive_distance(distance_m: float) -> None:
    """Проверить конечность и положительность расстояния в метрах."""
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Расстояние должно быть положительным и конечным, в метрах.")


def _input_factor(
    u_m: float | NDArray[np.float64],
    v_m: float | NDArray[np.float64],
    distance_m: float | NDArray[np.float64],
) -> np.complex128 | NDArray[np.complex128]:
    """Входной множитель Френеля для фиксированных координат и дистанций."""
    return np.exp(1j * PI * (u_m**2 + v_m**2) / (WAVELENGTH_M * distance_m))


def _output_factor(
    x_m: float | NDArray[np.float64],
    y_m: float | NDArray[np.float64],
    distance_m: float | NDArray[np.float64],
) -> np.complex128 | NDArray[np.complex128]:
    """Выходной множитель, включая exp(ikz)/(i*λ*z) и площадь пикселя ПВМС."""
    prefactor = np.exp(1j * WAVE_NUMBER_PER_M * distance_m) / (1j * WAVELENGTH_M * distance_m)
    output_chirp = _input_factor(x_m, y_m, distance_m)
    return prefactor * output_chirp * PIXEL_PITCH_M**2


def phase_to_field(phase_map: tf.Tensor, amplitude: float = DEFAULT_AMPLITUDE) -> tf.Tensor:
    """Преобразовать фазу произвольной формы в H = A*exp(i*φ), сохранив градиент.

    Вход — float32/float64, фаза в [−π, π] рад; выход — complex64/complex128
    той же формы. Амплитуда A конечна, неотрицательна и одинакова по полю.
    """
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Амплитуда должна быть конечной и неотрицательной.")
    phase_map = tf.convert_to_tensor(phase_map)
    if phase_map.dtype not in (tf.float32, tf.float64):
        raise ValueError("Фаза должна иметь тип float32 или float64.")
    tf.debugging.assert_all_finite(phase_map, "Фаза содержит NaN/Inf.")
    tf.debugging.assert_greater_equal(phase_map, tf.constant(PHASE_MIN_RAD, phase_map.dtype))
    tf.debugging.assert_less_equal(phase_map, tf.constant(PHASE_MAX_RAD, phase_map.dtype))
    unit_field = tf.complex(tf.cos(phase_map), tf.sin(phase_map))
    return unit_field * tf.constant(amplitude, unit_field.dtype)


def field_intensity(output_field: tf.Tensor) -> tf.Tensor:
    """Вернуть I = |K|² той же формы, float32/float64, без нормировки яркости."""
    return tf.square(tf.math.real(output_field)) + tf.square(tf.math.imag(output_field))


class FresnelPropagation(tf.Module):
    """Фиксированная физика: фаза B×128×128 → интенсивности B×P×128×128.

    По умолчанию P = 4, расстояния — 24,1 / 25,3 / 26,5 / 27,7 мм.
    Одна фаза каждого примера используется для всех плоскостей. Порядок
    осей результата: батч, плоскость, y, x. Обучаемых параметров нет.
    dtype задаёт float32/complex64 для обучения либо float64/complex128
    для точного восстановления. Шаг выходной сетки равен λ*z/(N*Δu).
    """

    def __init__(
        self,
        distances_m: tuple[float, ...] = RECONSTRUCTION_DISTANCES_M,
        dtype: tf.dtypes.DType = tf.float32,
    ) -> None:
        super().__init__(name="fresnel_propagation")
        self.dtype = tf.as_dtype(dtype)
        if self.dtype not in (tf.float32, tf.float64):
            raise ValueError("Точность расчёта должна быть float32 или float64.")
        self.complex_dtype = tf.complex64 if self.dtype == tf.float32 else tf.complex128
        if not distances_m:
            raise ValueError("Нужно хотя бы одно расстояние восстановления.")
        for distance_m in distances_m:
            _positive_distance(distance_m)

        # Коэффициенты не зависят от фазы: NumPy используется только здесь,
        # до построения обучающего графа, с явным переводом констант в TensorFlow.
        distances = np.asarray(distances_m, dtype=np.float64)
        indices = np.arange(IMAGE_SIZE, dtype=np.float64) - IMAGE_SIZE // 2
        input_axis_m = indices * PIXEL_PITCH_M
        input_chirp = _input_factor(
            input_axis_m[None, None, :],
            input_axis_m[None, :, None],
            distances[:, None, None],
        )
        output_pitch_m = WAVELENGTH_M * distances / (IMAGE_SIZE * PIXEL_PITCH_M)
        output_axis_m = output_pitch_m[:, None] * indices[None, :]
        output_factor = _output_factor(
            output_axis_m[:, None, :],
            output_axis_m[:, :, None],
            distances[:, None, None],
        )
        self.output_pixel_pitch_m = tf.constant(output_pitch_m, tf.float64)
        self._input_chirp = tf.constant(input_chirp, self.complex_dtype)
        self._output_factor = tf.constant(output_factor, self.complex_dtype)

    def _centered_spectrum(self, slm_field: tf.Tensor) -> tf.Tensor:
        """Применить входной chirp и один прямой 2D FFT на плоскость."""
        prepared_field = slm_field[:, None] * self._input_chirp[None]
        return tf.signal.fftshift(
            tf.signal.fft2d(tf.signal.ifftshift(prepared_field, axes=(-2, -1))),
            axes=(-2, -1),
        )

    def propagate(self, slm_field: tf.Tensor) -> tf.Tensor:
        """Распространить комплексное H формы B×128×128 в K_z формы B×P×128×128.

        K_z = exp(ikz)/(i*λ*z) * exp(i*π*(x²+y²)/(λ*z)) * Δu²
              * FFT2[H(u,v) * exp(i*π*(u²+v²)/(λ*z))].
        Прямое FFT не нормируется. Один батчевый вызов fft2d выполняет
        отдельный 2D FFT каждой пары «пример — плоскость». Амплитуда H
        может быть неоднородной; входное поле не изменяется.
        """
        slm_field = tf.cast(slm_field, self.complex_dtype)
        slm_field = tf.ensure_shape(slm_field, (None, *IMAGE_SHAPE))
        tf.debugging.assert_all_finite(tf.math.real(slm_field), "Поле H содержит NaN/Inf.")
        tf.debugging.assert_all_finite(tf.math.imag(slm_field), "Поле H содержит NaN/Inf.")
        return self._centered_spectrum(slm_field) * self._output_factor[None]

    def field(self, phase_map: tf.Tensor) -> tf.Tensor:
        """Получить комплексное K_z формы B×P×128×128 из фаз B×128×128 в радианах."""
        phase_map = tf.cast(phase_map, self.dtype)
        phase_map = tf.ensure_shape(phase_map, (None, *IMAGE_SHAPE))
        return self.propagate(phase_to_field(phase_map))

    def unitary_propagate(self, slm_field: tf.Tensor) -> tf.Tensor:
        """Вернуть U(H), B×P×128×128, для амплитудных проекций.

        Использует тот же входной chirp и один FFT на плоскость, что propagate.
        FFT делится на N; выходной chirp и коэффициент Френеля опущены.
        Они не меняют интенсивность после нормировки каждой плоскости.
        """
        slm_field = tf.ensure_shape(tf.cast(slm_field, self.complex_dtype), (None, *IMAGE_SHAPE))
        return self._centered_spectrum(slm_field) / tf.cast(IMAGE_SIZE, self.complex_dtype)

    def unitary_adjoint(self, output_field: tf.Tensor) -> tf.Tensor:
        """Вернуть U*(K), B×128×128: сумма сопряжённых операторов плоскостей.

        Вход K имеет форму B×P×128×128. Это сопряжённый оператор
        для скалярного произведения, а не обратный полного Fresnel.
        Для одновременной проекции результат дополнительно делится на P.
        """
        output_field = tf.ensure_shape(
            tf.cast(output_field, self.complex_dtype),
            (None, self._input_chirp.shape[0], *IMAGE_SHAPE),
        )
        backward = tf.signal.fftshift(
            tf.signal.ifft2d(tf.signal.ifftshift(output_field, axes=(-2, -1))),
            axes=(-2, -1),
        ) * tf.cast(IMAGE_SIZE, self.complex_dtype)
        return tf.reduce_sum(backward * tf.math.conj(self._input_chirp[None]), axis=1)

    def __call__(self, phase_map: tf.Tensor) -> tf.Tensor:
        """Вернуть физическую интенсивность B×P×128×128 без нормировки и квантования."""
        return field_intensity(self.field(phase_map))


def slm_field(phase: float, amplitude: float = DEFAULT_AMPLITUDE) -> complex:
    """Вернуть Python complex H = A*exp(i*φ) одного пикселя; фаза в [−π, π] рад."""
    if not PHASE_MIN_RAD <= phase <= PHASE_MAX_RAD:
        raise ValueError("Фаза должна находиться в диапазоне [−π, π].")
    return complex(phase_to_field(tf.constant(phase, tf.float64), amplitude).numpy())


def slm_field_matrix(
    phases: NDArray[np.floating], amplitude: float = DEFAULT_AMPLITUDE
) -> NDArray[np.complex128]:
    """Собрать H 128×128, complex128, через общий phase_to_field.

    phases — вещественная матрица 128×128 в [−π, π] рад. По умолчанию A = 1.
    Для финальных реконструкций фаза декодируется из сохранённого PNG ДОЭ.
    Адаптер NumPy вызывается вне обучения; исходная матрица не изменяется.
    """
    phases = np.asarray(phases)
    if phases.shape != IMAGE_SHAPE:
        raise ValueError("Фазовая карта должна иметь размер 128×128.")
    if phases.dtype.kind not in "fiu":
        raise ValueError("Фазовая карта должна содержать вещественные числа в радианах.")
    phases = phases.astype(np.float64)
    if not np.isfinite(phases).all() or np.any((phases < PHASE_MIN_RAD) | (phases > PHASE_MAX_RAD)):
        raise ValueError("Фаза должна находиться в диапазоне [−π, π].")
    return phase_to_field(tf.constant(phases, tf.float64), amplitude).numpy()


def fresnel_input(field: complex, u_m: float, v_m: float, distance_m: float) -> complex:
    """Подготовить один пиксель H к FFT общим входным множителем; все длины в СИ."""
    _positive_distance(distance_m)
    return complex(field * _input_factor(u_m, v_m, distance_m))


def fresnel_propagate(fft_value: complex, x_m: float, y_m: float, distance_m: float) -> complex:
    """Получить K одного пикселя по коэффициенту центрированного FFT; длины в СИ.

    Использует общий выходной множитель Френеля с площадью входного пикселя.
    x_m, y_m задаются на сетке с шагом λ*z/(128*PIXEL_PITCH_M).
    """
    _positive_distance(distance_m)
    return complex(fft_value * _output_factor(x_m, y_m, distance_m))


def _propagate_matrix(field: NDArray[np.complexfloating], distance_m: float) -> tf.Tensor:
    """Проверить NumPy поле H 128×128 и получить тензор K с complex128."""
    field = np.asarray(field)
    if field.shape != IMAGE_SHAPE:
        raise ValueError("Поле H должно иметь размер 128×128.")
    if field.dtype.kind not in "fciu" or not np.isfinite(field).all():
        raise ValueError("Поле H должно содержать конечные вещественные или комплексные числа.")
    propagation = FresnelPropagation(distances_m=(distance_m,), dtype=tf.float64)
    return propagation.propagate(tf.constant(field[None], tf.complex128))[0, 0]


def fresnel_field_matrix(
    field: NDArray[np.complexfloating], distance_m: float
) -> NDArray[np.complex128]:
    """Рассчитать K 128×128 через общий TensorFlow тракт с complex128.

    Вход — конечное поле H 128×128; distance_m — положительная дистанция, м.
    Один центрированный 2D FFT, выходной шаг λ*z/(128*PIXEL_PITCH_M).
    NumPy используется только на границах адаптера, вне обучения.
    ValueError: неверная форма, неконечные / нечисловые данные или z <= 0.
    """
    return _propagate_matrix(field, distance_m).numpy()


def reconstruct_intensity(
    field: NDArray[np.complexfloating], distance_m: float
) -> NDArray[np.float64]:
    """Вернуть I_z = |K_z|², float64 128×128, без нормировки и квантования.

    Адаптер общего расчёта для одного поля H и одного расстояния в метрах.
    Для финальных результатов H собирают из фазы сохранённого PNG ДОЭ.
    """
    return field_intensity(_propagate_matrix(field, distance_m)).numpy()
