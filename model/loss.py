"""Дифференцируемые потери по амплитуде, Pearson и глобальному SSIM."""

import tensorflow as tf

from utils.const import IMAGE_SIZE, INPUT_IMAGE_COUNT, MAX_PIXEL_VALUE, SSIM_C1, SSIM_C2
from utils.mathlib import field_intensity

AMPLITUDE_EPSILON = 1e-8
PEARSON_VARIANCE_FLOOR = 1e-12
MIXED_METRIC_WEIGHT = 0.25
LOSS_MODES = ("amplitude", "mixed")
SPATIAL_AXES = (-2, -1)


def target_amplitude(targets: tf.Tensor) -> tf.Tensor:
    """Проверить B×4×128×128 в [0,1] и вернуть sqrt(T/mean(T)+eps).

    Среднее берётся отдельно по каждой плоскости. Полностью чёрная
    плоскость отвергается: положительное среднее — допущение этой модели,
    а не условие ТЗ. Возвращается float32 на том же устройстве.
    """
    targets = tf.ensure_shape(
        tf.cast(targets, tf.float32), (None, INPUT_IMAGE_COUNT, IMAGE_SIZE, IMAGE_SIZE)
    )
    tf.debugging.assert_positive(tf.shape(targets)[0], message="Пустой батч целей.")
    tf.debugging.assert_all_finite(targets, "Цели содержат NaN/Inf.")
    tf.debugging.assert_greater_equal(targets, 0.0, message="Цели должны быть в [0,1].")
    tf.debugging.assert_less_equal(targets, 1.0, message="Цели должны быть в [0,1].")
    means = tf.reduce_mean(targets, axis=(-2, -1), keepdims=True)
    tf.debugging.assert_positive(
        means, message="Полностью чёрная плоскость: T/mean(T) требует положительного среднего."
    )
    return tf.sqrt(targets / means + AMPLITUDE_EPSILON)


def amplitude_loss(targets: tf.Tensor, unitary_field: tf.Tensor) -> tf.Tensor:
    """MSE амплитуд B×4×128×128 при унитарном FFT и освещении A=1.

    По Парсевалю средняя |U(exp(iφ))|² = 1 в каждой плоскости.
    Выходные множители полного Fresnel сокращаются при нормировке I/mean(I).
    PNG-квантование применяется только при экспорте и расчёте метрик.
    """
    amplitude = target_amplitude(targets)
    tf.debugging.assert_equal(tf.shape(amplitude), tf.shape(unitary_field))
    return tf.reduce_mean(tf.square(tf.abs(unitary_field) - amplitude))


def _intensity_moments(
    targets: tf.Tensor, reconstructed: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    target_mean = tf.reduce_mean(targets, axis=SPATIAL_AXES, keepdims=True)
    reconstructed_mean = tf.reduce_mean(reconstructed, axis=SPATIAL_AXES, keepdims=True)
    target_centered = targets - target_mean
    reconstructed_centered = reconstructed - reconstructed_mean
    return (
        tf.squeeze(target_mean, axis=SPATIAL_AXES),
        tf.squeeze(reconstructed_mean, axis=SPATIAL_AXES),
        tf.reduce_mean(tf.square(target_centered), axis=SPATIAL_AXES),
        tf.reduce_mean(tf.square(reconstructed_centered), axis=SPATIAL_AXES),
        tf.reduce_mean(target_centered * reconstructed_centered, axis=SPATIAL_AXES),
    )


def _nonconstant(images: tf.Tensor) -> tf.Tensor:
    return tf.reduce_min(images, axis=SPATIAL_AXES) != tf.reduce_max(images, axis=SPATIAL_AXES)


def intensity_pearson(
    targets: tf.Tensor, reconstructed: tf.Tensor, *, exact: bool = False
) -> tuple[tf.Tensor, tf.Tensor]:
    """Вернуть Pearson и маску учитываемых пар B×4 для интенсивностей B×4×H×W.

    В обучении постоянные цели исключаются, а нижняя граница знаменателя
    сохраняет конечные градиенты даже при постоянной реконструкции.
    Для отчёта exact исключает любую постоянную пару и убирает границу.
    """
    _, _, target_variance, reconstruction_variance, covariance = _intensity_moments(
        targets, reconstructed
    )
    valid = _nonconstant(targets)
    variance_product = target_variance * reconstruction_variance
    if exact:
        valid = valid & _nonconstant(reconstructed) & (variance_product > 0)
    else:
        variance_product = tf.maximum(variance_product, PEARSON_VARIANCE_FLOOR)
    denominator = tf.sqrt(tf.where(valid, variance_product, tf.ones_like(variance_product)))
    correlation = tf.where(valid, covariance / denominator, tf.zeros_like(covariance))
    return tf.clip_by_value(correlation, -1.0, 1.0), valid


def global_ssim(targets: tf.Tensor, reconstructed: tf.Tensor) -> tf.Tensor:
    """Вернуть глобальный SSIM B×4 для интенсивностей B×4×H×W в [0, 1]."""
    target_mean, reconstructed_mean, target_variance, reconstruction_variance, covariance = (
        _intensity_moments(targets, reconstructed)
    )
    luminance_constant = SSIM_C1 / MAX_PIXEL_VALUE**2
    structure_constant = SSIM_C2 / MAX_PIXEL_VALUE**2
    luminance = (2 * target_mean * reconstructed_mean + luminance_constant) / (
        tf.square(target_mean) + tf.square(reconstructed_mean) + luminance_constant
    )
    structure = (2 * covariance + structure_constant) / (
        target_variance + reconstruction_variance + structure_constant
    )
    return luminance * structure


def mixed_loss(targets: tf.Tensor, unitary_field: tf.Tensor) -> tf.Tensor:
    """Сложить MSE амплитуд и штрафы Pearson и SSIM по четырём плоскостям.

    Использует непрерывное поле B×4×128×128; SSIM сравнивает интенсивности
    после нормировки реконструкций по максимуму каждой плоскости.
    """
    targets = tf.cast(targets, tf.float32)
    amplitude_error = amplitude_loss(targets, unitary_field)
    intensity = field_intensity(unitary_field)
    pearson, valid = intensity_pearson(targets, intensity)
    valid_pairs = tf.cast(valid, pearson.dtype)
    pearson_error = tf.math.divide_no_nan(
        tf.reduce_sum(valid_pairs - pearson), tf.reduce_sum(valid_pairs)
    )
    ssim_error = tf.reduce_mean(1 - global_ssim(targets, normalize_intensity(intensity)))
    return amplitude_error + MIXED_METRIC_WEIGHT * (pearson_error + ssim_error)


def normalize_intensity(intensity: tf.Tensor) -> tf.Tensor:
    """Нормировать каждую плоскость B×4×128×128 по собственному максимуму.

    Возвращает float32 в [0, 1], нулевое изображение остаётся нулевым.
    Это та же нормировка I/max(I), что в pictures.OutputPictures, без
    округления в uint8. Минимум не вычитается, гамма-коррекции нет.
    """
    intensity = tf.ensure_shape(
        tf.cast(intensity, tf.float32), (None, INPUT_IMAGE_COUNT, IMAGE_SIZE, IMAGE_SIZE)
    )
    peaks = tf.reduce_max(intensity, axis=(-2, -1), keepdims=True)
    return tf.math.divide_no_nan(intensity, peaks)


def physics_loss(target_intensity: tf.Tensor, reconstructed_intensity: tf.Tensor) -> tf.Tensor:
    """Вернуть среднюю MSE с равными весами батчей, плоскостей и пикселей.

    Оба входа имеют форму B×4×128×128 и тип float32. Цели — исходные
    пиксели, делённые на 255, в [0, 1]. Реконструкции — физические |K_z|²;
    перед сравнением каждая делится на свой максимум. Потеря не оценивает
    абсолютную оптическую яркость. Квантование применяется при экспорте
    ДОЭ; итоговые метрики считают после загрузки этого файла.
    """
    target_intensity = tf.ensure_shape(
        tf.cast(target_intensity, tf.float32), (None, INPUT_IMAGE_COUNT, IMAGE_SIZE, IMAGE_SIZE)
    )
    normalized = normalize_intensity(reconstructed_intensity)
    tf.debugging.assert_equal(
        tf.shape(target_intensity),
        tf.shape(normalized),
        message="Формы целей и реконструкций должны совпадать.",
    )
    return tf.reduce_mean(tf.square(normalized - target_intensity))
