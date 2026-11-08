"""Обучение на train, выбор по validation и однократная test-оценка."""

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from model.dataset import TrainingPictures
from model.loss import LOSS_MODES, MIXED_METRIC_WEIGHT
from model.model import WARM_START_TARGET_STAGES, Model
from model.unrolled import DEFAULT_INITIAL_PHASE_SEED, DEFAULT_STAGES
from utils import const
from utils.evaluation import SavedDoeEvaluator
from utils.pictures import normalize_pixels, save_phase


def write_json(path: Path, contents: Any) -> None:
    """Сохранить строгий JSON; неопределённые метрики должны быть None, не NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _defined(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _mean_defined(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _output_correlation(first: np.ndarray, other: np.ndarray) -> float | None:
    first = first.astype(np.float64).ravel()
    other = other.astype(np.float64).ravel()
    first -= first.mean()
    other -= other.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(other)
    return _defined(float(np.dot(first, other) / denominator)) if denominator else None


def evaluate_model(
    model: Model,
    pictures: TrainingPictures,
    output_directory: Path,
    *,
    learned: bool = True,
    split_name: str = "test",
) -> dict[str, Any]:
    """Сохранить один ДОЭ на четвёрку, восстановить его и записать 8 метрик.

    Протокол: reload 8-bit DOE → полный Fresnel float64 → I/max(I) → PNG
    → reload → Pearson и глобальный SSIM в [0,255]. Неопределённый Pearson
    сохраняется как null и исключается из среднего с явным числом пар.
    Никаких обновлений весов или отдельной оптимизации фазы здесь нет.
    """
    output_directory = Path(output_directory)
    if not pictures:
        logging.info("Выборка %s пуста: средние метрики не рассчитаны.", split_name)
        return {"count": 0, "samples": []}
    metric_sums = np.zeros((const.INPUT_IMAGE_COUNT, 2), dtype=np.float64)
    metric_counts = np.zeros((const.INPUT_IMAGE_COUNT, 2), dtype=np.int64)
    evaluator = SavedDoeEvaluator()
    results = []
    correlations: list[float | None] = []
    phase_distances = []
    first_pixels, first_phase = None, None
    for sample_index, quartet in enumerate(pictures.quartets):
        sample_directory = quartet[0].parent
        target_pixels = pictures.read_quartet(sample_index)
        images = normalize_pixels(target_pixels)[None]
        phase_map = (
            model.predict_phase(images) if learned else model.predict_phase(images, learned=False)
        )[0].numpy()
        sample_output = output_directory / sample_directory.name
        doe_path = sample_output / "doe.png"
        save_phase(doe_path, phase_map)
        evaluation = evaluator.evaluate(doe_path, sample_output / "reconstructions", target_pixels)
        decoded = evaluation.phase
        reconstructed_pixels = evaluation.reconstructions
        for plane in evaluation.planes:
            index = plane["plane"]
            pearson, ssim = plane["pearson"], plane["ssim"]
            if pearson is not None:
                metric_sums[index - 1, 0] += pearson
                metric_counts[index - 1, 0] += 1
            metric_sums[index - 1, 1] += ssim
            metric_counts[index - 1, 1] += 1
            logging.info(
                "%s / %d.png (z=%.1f мм): Pearson=%s, SSIM=%.6f",
                sample_directory.name,
                index,
                plane["distance_m"] * 1000,
                f"{pearson:.6f}" if pearson is not None else "не определён",
                ssim,
            )
        if first_pixels is None:
            first_pixels, first_phase = reconstructed_pixels, decoded
        else:
            correlations.extend(
                _output_correlation(first, current)
                for first, current in zip(first_pixels, reconstructed_pixels)
            )
            phase_distances.append(
                float(
                    np.sqrt(np.mean(np.abs(np.exp(1j * decoded) - np.exp(1j * first_phase)) ** 2))
                )
            )
        sample_metrics = {
            "sample": sample_directory.name,
            "planes": evaluation.planes,
            "diagnostics": {
                "phase_edge_fraction": float(np.mean(np.abs(phase_map) > 0.99 * np.pi)),
                "target_spatial_std": float(np.std(target_pixels / 255, axis=(-2, -1)).mean()),
                "reconstruction_spatial_std": float(
                    np.std(reconstructed_pixels / 255, axis=(-2, -1)).mean()
                ),
            },
        }
        write_json(sample_output / "metrics.json", sample_metrics)
        results.append(sample_metrics)
        logging.info("Восстановлен пример %s: %s", split_name, sample_directory.name)

    labels = [f"z={distance * 1000:.1f} мм" for distance in const.RECONSTRUCTION_DISTANCES_M]
    labels.append("все плоскости")
    sums = np.vstack((metric_sums, metric_sums.sum(axis=0)))
    counts = np.vstack((metric_counts, metric_counts.sum(axis=0)))
    means = [
        (float(total[0] / count[0]) if count[0] else None, float(total[1] / count[1]))
        for total, count in zip(sums, counts, strict=True)
    ]
    for label, (pearson, ssim), count in zip(labels, means, counts, strict=True):
        pearson_text = f"{pearson:.6f}" if pearson is not None else "не определён"
        logging.info(
            "Средние метрики %s (%s): Pearson=%s (%d/%d пар), SSIM=%.6f",
            split_name,
            label,
            pearson_text,
            count[0],
            count[1],
            ssim,
        )
    summary = {
        "count": len(results),
        "mean_pearson": means[-1][0],
        "mean_ssim": means[-1][1],
        "per_plane_mean_pearson": [pearson for pearson, _ in means[:-1]],
        "per_plane_mean_ssim": [ssim for _, ssim in means[:-1]],
        "defined_metric_counts": metric_counts.tolist(),
        "diagnostics": {
            "phase_edge_fraction": float(
                np.mean([r["diagnostics"]["phase_edge_fraction"] for r in results])
            ),
            "output_correlation_to_first": _mean_defined(correlations),
            "phase_phasor_rms_to_first": float(np.mean(phase_distances))
            if phase_distances
            else None,
            "comparison_pairs": len(correlations),
        },
        "samples": results,
    }
    write_json(output_directory / "metrics.json", summary)
    return summary


def _write_history(output: Path, history: list[dict[str, Any]]) -> None:
    write_json(output / "history.json", history)
    columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "gradient_norm",
        "quartets",
        "batches",
        "best_epoch",
    ]
    if history and "val_score" in history[0]:
        columns.extend(("val_pearson", "val_ssim", "val_score", "val_pearson_pairs"))
    for split in ("train", "validation"):
        columns.extend(f"{split}_{name}" for name in ("pearson", "ssim", "phase_edge_fraction"))
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def train_model(
    data_directory: Path,
    output_directory: Path,
    epochs: int = 10,
    batch_size: int = 4,
    seed: int = 42,
    learning_rate: float = 1e-3,
    *,
    stages: int = DEFAULT_STAGES,
    initial_phase_seed: int = DEFAULT_INITIAL_PHASE_SEED,
    gradient_clip_norm: float = 1.0,
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
    monitor_count: int = 4,
    log_every: int = 10,
    init_model: Path | None = None,
    loss_mode: str = "amplitude",
) -> list[dict[str, Any]]:
    """Обучить с нуля или с переносом весов, выбрать по validation.

    По умолчанию используются полные split; подмножества задаются явно.
    Epoch monitoring использует фиксированные train/validation примеры.
    Amplitude выбирает checkpoint по loss, mixed — по среднему Pearson и
    SSIM на всём выбранном validation. Test открывается только
    после фиксации весов. Существующие результаты запуска не перезаписываются.
    При переносе 8 → 16 стадий untrained_test оценивает модель сразу после переноса.
    """
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"Неизвестный режим loss: {loss_mode}.")
    if epochs <= 0 or batch_size <= 0 or monitor_count < 0:
        raise ValueError("Эпохи и batch size должны быть положительными, monitor count >= 0.")
    if log_every <= 0:
        raise ValueError("Интервал вывода прогресса должен быть положительным.")
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, gradient_clip_norm)):
        raise ValueError(
            "Learning rate и gradient clip norm должны быть положительными и конечными."
        )
    for count in (train_count, validation_count, test_count):
        if count is not None and count <= 0:
            raise ValueError("Числа примеров должны быть положительными.")
    if init_model is not None and stages != WARM_START_TARGET_STAGES:
        raise ValueError(f"--init-model требует --stages {WARM_START_TARGET_STAGES}.")
    data_directory, output_directory = Path(data_directory), Path(output_directory)
    if any(
        (output_directory / name).exists() for name in ("model.keras", "config.json", "report.json")
    ):
        raise FileExistsError(f"В {output_directory} уже есть запуск. Укажите новую --output.")
    logging.info(
        "Подготовка обучения: data=%s, output=%s, устройство=CPU.", data_directory, output_directory
    )
    training = TrainingPictures(data_directory / "train", train_count)
    if not training:
        raise ValueError(f"В {data_directory / 'train'} нужен хотя бы один пример с 1.png–4.png.")
    validation_path = data_directory / "validation"
    validation = (
        TrainingPictures(validation_path, validation_count) if validation_path.exists() else None
    )
    if validation_count is not None and not validation:
        raise ValueError("Запрошена validation count, но validation отсутствует или пуст.")
    if not validation:
        if loss_mode == "mixed":
            raise ValueError("Режим mixed требует непустой validation.")
        validation = None
    monitored_splits = {
        name: split.prefix(min(monitor_count, len(split)))
        for name, split in (("train", training), ("validation", validation))
        if split and monitor_count
    }
    logging.info(
        "Train: %d четвёрок, %d пакетов/эпоху; validation: %d четвёрок, %d пакетов; эпох: %d.",
        len(training),
        math.ceil(len(training) / batch_size),
        len(validation) if validation else 0,
        math.ceil(len(validation) / batch_size) if validation else 0,
        epochs,
    )
    with tf.device("/CPU:0"):
        logging.info("Создание сети.")
        model = Model(seed=seed, stages=stages, initial_phase_seed=initial_phase_seed)
        initialization = None
        if init_model is not None:
            init_model = Path(init_model).resolve()
            source_hash = hashlib.sha256(init_model.read_bytes()).hexdigest()
            source = Model.load(init_model, seed=seed)
            if hashlib.sha256(init_model.read_bytes()).hexdigest() != source_hash:
                raise ValueError("Исходный checkpoint изменился во время загрузки.")
            transfer = model.warm_start(source)
            initialization = {
                "path": str(init_model),
                "sha256": source_hash,
                **transfer,
            }
            logging.info(
                "Перенесены веса %d → %d стадий из %s.",
                transfer["source_stages"],
                stages,
                init_model,
            )
        initial_weights = model.network.get_weights()
        config = {
            "architecture": "LearnedUnrolled",
            "configuration": {
                "data": str(data_directory.resolve()),
                "output": str(output_directory.resolve()),
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "learning_rate": learning_rate,
                "stages": stages,
                "initial_phase_seed": initial_phase_seed,
                "gradient_clip_norm": gradient_clip_norm,
                "train_count": train_count,
                "validation_count": validation_count,
                "test_count": test_count,
                "monitor_count": monitor_count,
                "log_every": log_every,
                "init_model": initialization,
                "loss_mode": loss_mode,
            },
            "parameter_count": model.network.count_params(),
            "train_samples": training.manifest(),
            "validation_samples": validation.manifest() if validation else [],
            "physics": {
                "wavelength_m": const.WAVELENGTH_M,
                "pixel_pitch_m": const.PIXEL_PITCH_M,
                "distances_m": const.RECONSTRUCTION_DISTANCES_M,
                "illumination_amplitude": 1.0,
                "output_pixel_pitch_m": model.propagation.output_pixel_pitch_m.numpy().tolist(),
            },
            "loss": "MSE(abs(unitary Fresnel), sqrt(target/mean(target)+1e-8)); positive target means",
            "evaluation": "reload 8-bit DOE; full float64 Fresnel; reload round(255*I/max(I)) PNG; global SSIM",
            "selection": "minimum full validation loss including epoch 0"
            if validation
            else "last epoch; no validation",
            "environment": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "device": "CPU",
                "deterministic_ops": True,
                "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
                "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            },
        }
        if loss_mode == "mixed":
            config["loss"] += (
                f"; + {MIXED_METRIC_WEIGHT:g}*(1-mean intensity Pearson)"
                f" + {MIXED_METRIC_WEIGHT:g}*(1-mean global SSIM after per-plane peak normalization)"
            )
            config["selection"] = (
                "maximum (mean Pearson + mean global SSIM)/2 on full selected validation "
                "including epoch 0; equivalent in-memory 8-bit DOE and reconstruction rounding; "
                "exclude undefined Pearson pairs; ties favor earliest epoch"
            )
        write_json(output_directory / "config.json", config)
        observed_history = []

        def record_epoch(record: dict[str, Any]) -> None:
            for split_name, monitored in monitored_splits.items():
                logging.info(
                    "Эпоха %d: мониторинг %s на %d четвёрках.",
                    record["epoch"],
                    split_name,
                    len(monitored),
                )
                summary = evaluate_model(
                    model,
                    monitored,
                    output_directory / "monitoring" / f"epoch_{record['epoch']:04d}" / split_name,
                    split_name=f"{split_name} monitor",
                )
                record[f"{split_name}_pearson"] = summary["mean_pearson"]
                record[f"{split_name}_ssim"] = summary["mean_ssim"]
                record[f"{split_name}_phase_edge_fraction"] = summary["diagnostics"][
                    "phase_edge_fraction"
                ]
                record[f"{split_name}_monitor"] = summary
            observed_history.append(record)
            _write_history(output_directory, observed_history)

        history = model.train(
            training.as_dataset(batch_size=batch_size, seed=seed),
            epochs=epochs,
            learning_rate=learning_rate,
            validation_dataset=validation.as_dataset(batch_size=batch_size, shuffle=False)
            if validation
            else None,
            gradient_clip_norm=gradient_clip_norm,
            checkpoint_path=output_directory / "model.keras",
            on_epoch=record_epoch,
            log_every=log_every,
            loss_mode=loss_mode,
        )
        report = {
            **config,
            "best_epoch": model.best_epoch,
            "initial_validation_loss": model.initial_validation_loss,
            "best_validation_loss": model.best_validation_loss,
            "parameter_change_l2": float(
                np.sqrt(
                    sum(
                        np.sum((after - before).astype(np.float64) ** 2)
                        for after, before in zip(model.network.get_weights(), initial_weights)
                    )
                )
            ),
            "step_sizes": model.step_sizes(),
        }
        if loss_mode == "mixed":
            report["initial_validation_score"] = history[0]["val_score"]
            report["best_validation_score"] = history[model.best_epoch]["val_score"]
        # Freeze selection durably before discovering, loading or evaluating test.
        write_json(output_directory / "report.json", report)
        checkpoint = output_directory / "model.keras"
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        selected = Model.load(checkpoint, seed=seed)
        testing_path = data_directory / "test"
        testing = TrainingPictures(testing_path, test_count) if testing_path.exists() else None
        if test_count is not None and not testing:
            raise ValueError("Запрошена test count, но test отсутствует или пуст.")
        report["test_samples"] = testing.manifest() if testing else []
        if testing:
            report["trained_test"] = evaluate_model(selected, testing, output_directory / "test")
            report["untrained_test"] = evaluate_model(
                selected.with_weights(initial_weights),
                testing,
                output_directory / "untrained_test",
                split_name="warm start test" if initialization is not None else "untrained test",
            )
            report["classical_test"] = evaluate_model(
                selected,
                testing,
                output_directory / "classical_test",
                learned=False,
                split_name="classical test",
            )
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError("Checkpoint изменился при test-оценке.")
        report["checkpoint_sha256"] = checkpoint_hash
        write_json(output_directory / "report.json", report)
    logging.info("Выбрана эпоха %d; модель сохранена: %s", model.best_epoch, checkpoint)
    return history


def _publish_model(checkpoint: Path, destination: Path) -> None:
    """Заменить модель для инференса после полной записи выбранного checkpoint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-", dir=destination.parent) as directory:
        staged_checkpoint = Path(directory) / destination.name
        shutil.copyfile(checkpoint, staged_checkpoint)
        staged_checkpoint.replace(destination)
    logging.info("Модель для инференса сохранена: %s", destination)


def build_parser() -> argparse.ArgumentParser:
    """Создать CLI с параметрами, реально передаваемыми в обучение."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=const.DATASET_DIRECTORY, help="Папка набора данных"
    )
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--output",
        type=Path,
        default=const.PROJECT_ROOT / "runs" / f"learned_unrolled_{run_timestamp}",
        help="Папка запуска (по умолчанию runs/learned_unrolled_ГГГГММДД-ЧЧММСС).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--loss",
        choices=LOSS_MODES,
        default="amplitude",
        help="Амплитудная MSE или смешанная потеря с отбором по Pearson и SSIM на validation",
    )
    parser.add_argument("--stages", type=int, default=DEFAULT_STAGES)
    parser.add_argument("--initial-phase-seed", type=int, default=DEFAULT_INITIAL_PHASE_SEED)
    parser.add_argument(
        "--init-model", type=Path, help="Checkpoint с 8 стадиями для переноса весов на 16 стадий"
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}-count", type=int, default=None)
    parser.add_argument("--monitor-count", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Выводить прогресс каждые N пакетов (по умолчанию 10).",
    )
    return parser


def main() -> None:
    """Запустить обучение и опубликовать выбранную модель для инференса."""
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.threads <= 0:
        parser.error("--threads должен быть положительным.")
    if arguments.log_every <= 0:
        parser.error("--log-every должен быть положительным.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    tf.config.threading.set_intra_op_parallelism_threads(arguments.threads)
    tf.config.threading.set_inter_op_parallelism_threads(arguments.threads)
    try:
        if arguments.output.resolve() == const.DEFAULT_MODEL_PATH.parent.resolve():
            raise ValueError(
                "Для --output обучения укажите отдельный каталог, например runs/run_01."
            )
        train_model(
            data_directory=arguments.data,
            output_directory=arguments.output,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            learning_rate=arguments.learning_rate,
            stages=arguments.stages,
            initial_phase_seed=arguments.initial_phase_seed,
            gradient_clip_norm=arguments.gradient_clip_norm,
            train_count=arguments.train_count,
            validation_count=arguments.validation_count,
            test_count=arguments.test_count,
            monitor_count=arguments.monitor_count,
            log_every=arguments.log_every,
            init_model=arguments.init_model,
            loss_mode=arguments.loss,
        )
        _publish_model(arguments.output / "model.keras", const.DEFAULT_MODEL_PATH)
    except (ValueError, OSError, tf.errors.OpError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
