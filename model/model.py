"""Общие веса развёрнутой сети, обучение и выбор по validation."""

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from model.loss import (
    LOSS_MODES,
    SPATIAL_AXES,
    amplitude_loss,
    global_ssim,
    intensity_pearson,
    mixed_loss,
)
from model.unrolled import DEFAULT_INITIAL_PHASE_SEED, DEFAULT_STAGES, LearnedUnrolled
from utils import const
from utils.mathlib import FresnelPropagation

LOGGER = logging.getLogger(__name__)
EpochCallback = Callable[[dict[str, Any]], None]
WARM_START_SOURCE_STAGES = 8
WARM_START_TARGET_STAGES = 16
WARM_START_TAIL_LOGIT = -1.5


def _quantized_validation_planes(
    targets: tf.Tensor, phase_map: tf.Tensor, propagation: FresnelPropagation
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Вернуть Pearson, SSIM и маску определённого Pearson формы B×4.

    Расчёт B×4×128×128 эквивалентен PNG-кодировке фазы и реконструкций
    в pictures.py, включая float64 Fresnel и округление к ближайшему чётному.
    """
    phase_map = tf.cast(phase_map, tf.float64)
    pi = tf.constant(const.PI, tf.float64)
    max_pixel = const.MAX_PIXEL_VALUE
    phase_pixels = tf.clip_by_value(tf.round(max_pixel * (phase_map + pi) / (2 * pi)), 0, max_pixel)
    decoded_phase = phase_pixels / max_pixel * (2 * pi) - pi
    intensity = propagation(decoded_phase)
    peaks = tf.reduce_max(intensity, axis=SPATIAL_AXES, keepdims=True)
    normalized = tf.math.divide_no_nan(intensity, peaks)
    reconstructed = tf.clip_by_value(tf.round(max_pixel * normalized), 0, max_pixel) / max_pixel
    targets = tf.round(tf.cast(targets, tf.float64) * max_pixel) / max_pixel
    pearson, valid = intensity_pearson(targets, reconstructed, exact=True)
    ssim = global_ssim(targets, reconstructed)
    return pearson, ssim, valid


def _quantized_validation_metrics(
    targets: tf.Tensor, phase_map: tf.Tensor, propagation: FresnelPropagation
) -> tf.Tensor:
    """Вернуть суммы Pearson/SSIM и число определённых Pearson в батче."""
    pearson, ssim, valid = _quantized_validation_planes(targets, phase_map, propagation)
    return tf.stack(
        [tf.reduce_sum(pearson), tf.reduce_sum(ssim), tf.reduce_sum(tf.cast(valid, tf.float64))]
    )


def _log_progress(
    stage: str,
    batch: int,
    total_batches: int,
    samples: int,
    loss_sum: float,
    loss_mode: str,
    log_every: int,
) -> None:
    if batch == 1 or batch % log_every == 0 or batch == total_batches:
        LOGGER.info(
            "%s: пакет %d/%s, четвёрок=%d, %s loss=%.6f",
            stage,
            batch,
            total_batches if total_batches >= 0 else "?",
            samples,
            loss_mode,
            loss_sum / samples,
        )


def _run_training_epoch(
    dataset: tf.data.Dataset,
    step: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    stage: str,
    loss_mode: str,
    log_every: int,
) -> dict[str, float | int]:
    """Обновить веса на каждом батче и усреднить статистики по четвёркам."""
    LOGGER.info("%s: начало; первый пакет может включать подготовку графа TensorFlow.", stage)
    total_batches = int(tf.data.experimental.cardinality(dataset))
    loss_sum, norm_sum, sample_count, batch_count = 0.0, 0.0, 0, 0
    for batch_count, targets in enumerate(dataset, start=1):
        loss, norm = step(targets)
        batch_size = int(tf.shape(targets)[0])
        loss_sum += float(loss) * batch_size
        norm_sum += float(norm) * batch_size
        sample_count += batch_size
        _log_progress(
            stage, batch_count, total_batches, sample_count, loss_sum, loss_mode, log_every
        )
    if not sample_count:
        raise ValueError("Обучающий набор пуст.")
    return {
        "train_loss": loss_sum / sample_count,
        "gradient_norm": norm_sum / sample_count,
        "quartets": sample_count,
        "batches": batch_count,
    }


def _run_validation_epoch(
    dataset: tf.data.Dataset,
    step: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor | None]],
    stage: str,
    loss_mode: str,
    log_every: int,
) -> dict[str, float | int]:
    """Оценить полную validation без обновления весов и получить критерий отбора."""
    LOGGER.info("%s: начало; первый пакет может включать подготовку графа TensorFlow.", stage)
    total_batches = int(tf.data.experimental.cardinality(dataset))
    loss_sum, count = 0.0, 0
    pearson_sum, ssim_sum, pearson_pairs = 0.0, 0.0, 0
    for batch, targets in enumerate(dataset, start=1):
        batch_size = int(tf.shape(targets)[0])
        loss, metrics = step(targets)
        loss_sum += float(loss) * batch_size
        if metrics is not None:
            batch_pearson, batch_ssim, batch_pairs = metrics.numpy()
            pearson_sum += float(batch_pearson)
            ssim_sum += float(batch_ssim)
            pearson_pairs += int(batch_pairs)
        count += batch_size
        _log_progress(stage, batch, total_batches, count, loss_sum, loss_mode, log_every)
    if not count:
        raise ValueError("Валидационный набор пуст.")
    summary: dict[str, float | int] = {"val_loss": loss_sum / count}
    if loss_mode == "mixed":
        if not pearson_pairs:
            raise ValueError(
                "Режим mixed требует хотя бы одну пару validation с определённым Pearson."
            )
        pair_count = count * const.INPUT_IMAGE_COUNT
        pearson = pearson_sum / pearson_pairs
        ssim = ssim_sum / pair_count
        score = (pearson + ssim) / 2
        summary.update(
            val_pearson=pearson,
            val_ssim=ssim,
            val_score=score,
            val_pearson_pairs=pearson_pairs,
        )
        LOGGER.info(
            "%s: Pearson=%.6f (%d/%d пар), SSIM=%.6f, score=%.6f",
            stage,
            pearson,
            pearson_pairs,
            pair_count,
            ssim,
            score,
        )
    return summary


class Model:
    """Четыре цели B×4×128×128 → одна фаза B×128×128 в радианах.

    Восемь стадий и 961 обучаемый параметр по умолчанию. Все примеры
    используют одни веса, число стадий и фиксированную начальную фазу.
    train() создаёт новый Adam; восстановление его состояния не реализовано.
    """

    def __init__(
        self,
        seed: int = 42,
        stages: int = DEFAULT_STAGES,
        initial_phase_seed: int = DEFAULT_INITIAL_PHASE_SEED,
    ) -> None:
        if seed < 0:
            raise ValueError("Seed обучения должен быть неотрицательным.")
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        self.seed = seed
        self.network = LearnedUnrolled(
            stages=stages, initial_phase_seed=initial_phase_seed, name="four_images_to_one_doe"
        )
        self._predict = tf.function(self.network.__call__, reduce_retracing=True)
        self.propagation = FresnelPropagation()
        self.best_epoch = 0
        self.initial_validation_loss: float | None = None
        self.best_validation_loss: float | None = None

    def warm_start(self, source: "Model") -> dict[str, int | float]:
        """Перенести общую CNN и первые 8 шагов в 16-стадийную сеть."""
        destination = self.network
        if not isinstance(destination, LearnedUnrolled) or (
            destination.stages != WARM_START_TARGET_STAGES
        ):
            raise ValueError(f"--init-model требует --stages {WARM_START_TARGET_STAGES}.")
        original = source.network
        if not isinstance(original, LearnedUnrolled) or original.stages != WARM_START_SOURCE_STAGES:
            raise ValueError(
                f"--init-model требует checkpoint LearnedUnrolled с {WARM_START_SOURCE_STAGES} стадиями."
            )
        if original.initial_phase_seed != destination.initial_phase_seed:
            raise ValueError(
                "Для переноса укажите --initial-phase-seed "
                f"{original.initial_phase_seed}, как у исходного checkpoint."
            )
        tail_logits = tf.fill(
            (destination.stages - original.stages,),
            tf.cast(WARM_START_TAIL_LOGIT, destination.step_logits.dtype),
        )
        destination.correction.set_weights(original.correction.get_weights())
        source_logits = tf.cast(original.step_logits, destination.step_logits.dtype)
        destination.step_logits.assign(tf.concat([source_logits, tail_logits], axis=0))
        return {"source_stages": original.stages, "tail_logit": WARM_START_TAIL_LOGIT}

    def step_sizes(self) -> list[float]:
        """Вернуть значения обучаемых шагов для отчёта."""
        if not isinstance(self.network, LearnedUnrolled):
            raise ValueError("Обучаемые шаги доступны только для LearnedUnrolled.")
        return self.network.step_sizes().numpy().tolist()

    def with_weights(self, weights: list[NDArray[np.floating]]) -> "Model":
        """Создать отдельную сеть для baseline с теми же стадиями и начальной фазой."""
        if not isinstance(self.network, LearnedUnrolled):
            raise ValueError("Отдельный baseline доступен только для LearnedUnrolled.")
        clone = Model(
            seed=self.seed,
            stages=self.network.stages,
            initial_phase_seed=self.network.initial_phase_seed,
        )
        clone.network.set_weights(weights)
        return clone

    def predict_phase(self, images: tf.Tensor, learned: bool = True) -> tf.Tensor:
        """Синтезировать одну фазу на четвёрку без изменения общих весов."""
        images = tf.ensure_shape(
            tf.cast(images, tf.float32), (None, const.INPUT_IMAGE_COUNT, *const.IMAGE_SHAPE)
        )
        if isinstance(self.network, LearnedUnrolled):
            return self._predict(images, learned=learned)
        # Старые .keras остаются доступны для повторной оценки Dense checkpoint.
        if not learned:
            raise ValueError("Классический baseline доступен только для LearnedUnrolled.")
        return self._predict(images)

    def build_image(self, phase_map: tf.Tensor) -> tf.Tensor:
        """Рассчитать полные интенсивности B×4×128×128 из фаз B×128×128."""
        return self.propagation(phase_map)

    def objective(self, targets: tf.Tensor, *, loss_mode: str = "amplitude") -> tf.Tensor:
        """Вычислить непрерывную потерю общей сети без PNG-квантования."""
        if not isinstance(self.network, LearnedUnrolled):
            raise ValueError("Обучение старого Dense checkpoint этим методом не поддерживается.")
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"Неизвестный режим loss: {loss_mode}.")
        phase_map = self.network(targets)
        field = self.network.forward_field(phase_map)
        return (
            mixed_loss(targets, field) if loss_mode == "mixed" else amplitude_loss(targets, field)
        )

    def train(
        self,
        dataset: tf.data.Dataset,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        validation_dataset: tf.data.Dataset | None = None,
        gradient_clip_norm: float = 1.0,
        checkpoint_path: Path | None = None,
        on_epoch: EpochCallback | None = None,
        log_every: int = 10,
        *,
        loss_mode: str = "amplitude",
    ) -> list[dict[str, Any]]:
        """Обучить на train и восстановить лучшие веса, включая эпоху 0.

        Loss и норма градиента взвешены по числу четвёрок, включая последний
        неполный пакет. Amplitude выбирает минимум validation loss, а mixed
        требует validation и выбирает максимум среднего Pearson и SSIM после
        8-битного квантования. Amplitude без validation сохраняет последнюю эпоху.
        on_epoch вызывается после выбора checkpoint; он не обновляет веса.
        Прогресс выводится на первом, последнем и каждом log_every пакете.
        """
        if not isinstance(self.network, LearnedUnrolled):
            raise ValueError("Обучение старого Dense checkpoint этим методом не поддерживается.")
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"Неизвестный режим loss: {loss_mode}.")
        mixed = loss_mode == "mixed"
        if mixed and validation_dataset is None:
            raise ValueError("Режим mixed требует непустой validation.")
        if epochs <= 0:
            raise ValueError("Количество эпох должно быть положительным.")
        if log_every <= 0:
            raise ValueError("Интервал вывода прогресса должен быть положительным.")
        for name, value in (
            ("Learning rate", learning_rate),
            ("Gradient clip norm", gradient_clip_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} должен быть положительным и конечным.")
        for split in (dataset, validation_dataset):
            if split is not None and int(tf.data.experimental.cardinality(split)) == int(
                tf.data.INFINITE_CARDINALITY
            ):
                raise ValueError("Для эпох нужен конечный dataset, без repeat().")
        if int(tf.data.experimental.cardinality(dataset)) == 0:
            raise ValueError("Обучающий набор пуст.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        validation_propagation = FresnelPropagation(dtype=tf.float64) if mixed else None
        input_signature = [
            tf.TensorSpec((None, const.INPUT_IMAGE_COUNT, *const.IMAGE_SHAPE), tf.float32)
        ]

        @tf.function(input_signature=input_signature)
        def train_step(targets: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                loss = (
                    self.objective(targets, loss_mode="mixed") if mixed else self.objective(targets)
                )
            variables = self.network.trainable_variables
            gradients = tape.gradient(loss, variables)
            if any(gradient is None for gradient in gradients):
                raise ValueError("Не все обучаемые параметры связаны с loss.")
            norm = tf.linalg.global_norm(gradients)
            tf.debugging.assert_all_finite(loss, "Ошибка обучения содержит NaN/Inf.")
            tf.debugging.assert_all_finite(norm, "Градиент содержит NaN/Inf.")
            clipped, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
            optimizer.apply_gradients(zip(clipped, variables))
            return loss, norm

        @tf.function(input_signature=input_signature)
        def validation_step(targets: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor | None]:
            metrics = None
            if validation_propagation is None:
                loss = self.objective(targets)
            else:
                phase_map = self.network(targets)
                loss = mixed_loss(targets, self.network.forward_field(phase_map))
                metrics = _quantized_validation_metrics(targets, phase_map, validation_propagation)
            tf.debugging.assert_all_finite(loss, "Ошибка валидации содержит NaN/Inf.")
            return loss, metrics

        self.best_epoch = 0
        initial_record: dict[str, Any] = {"epoch": 0}
        if validation_dataset is not None:
            initial_record.update(
                _run_validation_epoch(
                    validation_dataset,
                    validation_step,
                    "Начальная валидация (эпоха 0)",
                    loss_mode,
                    log_every,
                )
            )
        else:
            LOGGER.warning("Validation не задан: будут сохранены веса последней эпохи.")
        self.initial_validation_loss = initial_record.get("val_loss")
        self.best_validation_loss = self.initial_validation_loss
        best_weights = self.network.get_weights()
        selection_key, direction = ("val_score", 1) if mixed else ("val_loss", -1)
        best_selection = direction * initial_record.get(selection_key, 0.0)
        history = [initial_record]
        if checkpoint_path is not None:
            self.save(checkpoint_path)
        if on_epoch is not None:
            on_epoch(initial_record)

        for epoch in range(1, epochs + 1):
            record: dict[str, Any] = {
                "epoch": epoch,
                **_run_training_epoch(
                    dataset, train_step, f"Эпоха {epoch}/{epochs}, обучение", loss_mode, log_every
                ),
            }
            if validation_dataset is not None:
                record.update(
                    _run_validation_epoch(
                        validation_dataset,
                        validation_step,
                        f"Эпоха {epoch}/{epochs}, валидация",
                        loss_mode,
                        log_every,
                    )
                )
            selection = direction * record.get(selection_key, 0.0)
            if validation_dataset is None or selection > best_selection:
                best_selection = selection
                self.best_epoch = epoch
                self.best_validation_loss = record.get("val_loss")
                best_weights = self.network.get_weights()
                if checkpoint_path is not None:
                    self.save(checkpoint_path)
            record["best_epoch"] = self.best_epoch
            history.append(record)
            LOGGER.info(
                "Эпоха %d/%d: четвёрок=%d, пакетов=%d, %s loss=%.6f, val=%s",
                epoch,
                epochs,
                record["quartets"],
                record["batches"],
                loss_mode,
                record["train_loss"],
                record.get("val_loss"),
            )
            if on_epoch is not None:
                on_epoch(record)
        self.network.set_weights(best_weights)
        return history

    def save(self, path: str | Path) -> None:
        """Сохранить архитектуру, seed начальной фазы и общие веса в .keras."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.network.save(path)

    @classmethod
    def load(cls, path: str | Path, seed: int = 42) -> "Model":
        """Загрузить сеть для синтеза; старые Dense .keras также читаются."""
        model = cls(seed=seed)
        model.network = tf.keras.models.load_model(Path(path), compile=False)
        model._predict = tf.function(model.network.__call__, reduce_retracing=True)
        return model
