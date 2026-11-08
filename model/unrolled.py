"""Обучаемая развёрнутая сеть из диагностического learned_unrolled/probe.py."""

from typing import Any

import numpy as np
import tensorflow as tf

from model.loss import target_amplitude
from utils import const
from utils.mathlib import FresnelPropagation

DEFAULT_STAGES = 8
DEFAULT_INITIAL_PHASE_SEED = 314159
CORRECTION_SCALE_RAD = 0.25


@tf.keras.utils.register_keras_serializable(package="doe")
class LearnedUnrolled(tf.keras.Model):
    """B×4×128×128 целей → одна общая фаза B×128×128 в [−π,π].

    Каждая стадия: прямой одно-FFT Fresnel, замена амплитуд, сопряжённая
    проекция четырёх полей и обучаемая круговая коррекция. Общая CNN
    4→8→8→1 сохраняет пространственные координаты. При нулевом последнем
    Conv и единичных шагах сеть совпадает с классической проекцией.
    Начальная фаза фиксирована для всех входов и не является параметром.
    """

    def __init__(
        self,
        stages: int = DEFAULT_STAGES,
        initial_phase_seed: int = DEFAULT_INITIAL_PHASE_SEED,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if stages <= 0:
            raise ValueError("Число стадий должно быть положительным.")
        if initial_phase_seed < 0:
            raise ValueError("Seed начальной фазы должен быть неотрицательным.")
        self.stages = stages
        self.initial_phase_seed = initial_phase_seed
        self.propagation = FresnelPropagation()
        initial_phase = np.random.default_rng(initial_phase_seed).uniform(
            -np.pi, np.pi, (1, *const.IMAGE_SHAPE)
        )
        self.initial_phase = tf.constant(initial_phase, tf.float32)
        self.step_logits = self.add_weight(name="step_logits", shape=(stages,), initializer="zeros")
        self.correction = tf.keras.Sequential(
            [
                tf.keras.Input((*const.IMAGE_SHAPE, 4)),
                tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer="zeros"),
            ],
            name="shared_phase_correction",
        )
        self.build((None, const.INPUT_IMAGE_COUNT, *const.IMAGE_SHAPE))

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """Пометить сеть построенной: CNN и коэффициенты созданы в __init__."""
        super().build(input_shape)

    def get_config(self) -> dict[str, Any]:
        """Сохранить стадии и seed вместе с архитектурой в .keras."""
        return {
            **super().get_config(),
            "stages": self.stages,
            "initial_phase_seed": self.initial_phase_seed,
        }

    def forward_field(self, phase_map: tf.Tensor) -> tf.Tensor:
        """Получить унитарные поля B×4×128×128, complex64, без разрыва графа."""
        field = tf.complex(tf.cos(phase_map), tf.sin(phase_map))
        return self.propagation.unitary_propagate(field)

    def step_sizes(self, stage: int | None = None) -> tf.Tensor:
        """Вернуть шаги развёртки или скалярный шаг заданной стадии."""
        logits = self.step_logits if stage is None else self.step_logits[stage]
        return 2 * tf.sigmoid(logits)

    def call(self, targets: tf.Tensor, learned: bool = True) -> tf.Tensor:
        """Выполнить фиксированное число стадий без подгонки входа на inference.

        learned=False отключает CNN и обучаемые шаги для отдельного
        классического baseline. Он не считается результатом обучения.
        """
        amplitude = target_amplitude(targets)
        phase_map = tf.broadcast_to(self.initial_phase, (tf.shape(targets)[0], *const.IMAGE_SHAPE))
        for stage in range(self.stages):
            field_phase = tf.math.angle(self.forward_field(phase_map))
            projected = tf.cast(amplitude, tf.complex64) * tf.complex(
                tf.cos(field_phase), tf.sin(field_phase)
            )
            back = self.propagation.unitary_adjoint(projected) / const.INPUT_IMAGE_COUNT
            current = tf.complex(tf.cos(phase_map), tf.sin(phase_map))
            delta = tf.math.angle(back * tf.math.conj(current))
            if learned:
                features = tf.stack(
                    [tf.sin(delta), tf.cos(delta), tf.abs(back), tf.abs(back) ** 2], axis=-1
                )
                correction = CORRECTION_SCALE_RAD * tf.tanh(self.correction(features)[..., 0])
                phase_map += self.step_sizes(stage) * delta + correction
            else:
                phase_map += delta
        phase_map = tf.math.angle(tf.complex(tf.cos(phase_map), tf.sin(phase_map)))
        tf.debugging.assert_all_finite(phase_map, "Фаза сети содержит NaN/Inf.")
        return phase_map
