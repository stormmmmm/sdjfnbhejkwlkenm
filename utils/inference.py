"""Синтез одного фазового ДОЭ на каждый комплект из четырёх PNG."""

import argparse
import hashlib
import json
import logging
import platform
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np
import PIL
import tensorflow as tf
from numpy.typing import NDArray

from model.model import Model
from model.unrolled import LearnedUnrolled
from utils import const
from utils.const import DEFAULT_MODEL_PATH, PROJECT_ROOT
from utils.evaluation import SavedDoeEvaluator
from utils.pictures import load_grayscale_png, normalize_pixels, save_phase

DEFAULT_INPUT = PROJECT_ROOT / "input"
DEFAULT_OUTPUT = DEFAULT_MODEL_PATH.parent
DEFAULT_MODEL = DEFAULT_MODEL_PATH
PACKAGED_MODEL = PROJECT_ROOT / "weights" / "model.keras"
PLANE_PNG_NAMES = tuple(f"{index}.png" for index in range(1, const.INPUT_IMAGE_COUNT + 1))
RESULT_NAMES = ("doe.png", "reconstructions", "inputs", "metrics.json", "run.json")
FLAT_RESULT_NAMES = ("doe.png", *PLANE_PNG_NAMES, "inputs", "metrics.json", "run.json")
SUMMARY_NAME = "summary.json"
RESULT_LOCK_NAME = ".synthesis-lock"
LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, contents: Any) -> None:
    path.write_text(
        json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _check_directories(input_directory: Path, output_directory: Path) -> None:
    if (
        input_directory == output_directory
        or input_directory in output_directory.parents
        or output_directory in input_directory.parents
    ):
        raise ValueError("Папки входа и результатов не должны совпадать или содержать друг друга.")
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Нет папки входа: {input_directory}.")


def _check_destination(output_directory: Path, result_names: tuple[str, ...]) -> None:
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"Путь результатов должен быть папкой: {output_directory}")
    occupied = [
        name for name in result_names if (output_directory / name).exists(follow_symlinks=False)
    ]
    if occupied:
        raise FileExistsError(
            f"В {output_directory} уже есть результаты: {', '.join(occupied)}. "
            "Для повторного запуска укажите новую папку --output."
        )


@contextmanager
def _staged_publication(output_directory: Path, result_names: tuple[str, ...]) -> Iterator[Path]:
    """Подготовить результат под блокировкой и опубликовать файлы в заданном порядке."""
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_directory = output_directory / RESULT_LOCK_NAME
    try:
        lock_directory.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"В {output_directory} уже выполняется публикация ДОЭ.") from error
    try:
        _check_destination(output_directory, result_names)
        with tempfile.TemporaryDirectory(prefix=".synthesis-", dir=output_directory) as directory:
            staging = Path(directory)
            yield staging
            published = []
            try:
                for name in result_names:
                    (staging / name).rename(output_directory / name)
                    published.append(name)
            except BaseException:
                for name in reversed(published):
                    (output_directory / name).replace(staging / name)
                raise
    finally:
        lock_directory.rmdir()


def _load_input_snapshot(
    input_directory: Path, names: set[str]
) -> tuple[NDArray[np.uint8], dict[str, bytes]]:
    """Загрузить и сохранить в памяти одни и те же исходные байты PNG."""
    contents = {name: (input_directory / name).read_bytes() for name in sorted(names)}
    pixels = np.stack(
        [
            load_grayscale_png(input_directory / name, contents=image_bytes)
            for name, image_bytes in contents.items()
        ]
    )
    return pixels, contents


def _load_model_snapshot(model_path: Path, seed: int) -> tuple[Model, str]:
    """Загрузить .keras из снимка: его хеш точно соответствует весам расчёта."""
    contents = model_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="doe-checkpoint-") as directory:
        snapshot = Path(directory) / "model.keras"
        snapshot.write_bytes(contents)
        try:
            model = Model.load(snapshot, seed=seed)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Не удалось загрузить checkpoint {model_path}: {error}") from error
    return model, hashlib.sha256(contents).hexdigest()


def _resolve_model_path(model_path: str | Path | None) -> Path:
    """Выбрать явный checkpoint, локальные веса или поставляемые веса."""
    if model_path is not None:
        return Path(model_path).resolve()
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL.resolve()
    return PACKAGED_MODEL.resolve()


def synthesize(
    input_directory: str | Path = DEFAULT_INPUT,
    output_directory: str | Path = DEFAULT_OUTPUT,
    model_path: str | Path | None = None,
    *,
    seed: int = 42,
    flat_reconstructions: bool = False,
) -> dict[str, Any]:
    """Сохранить один ДОЭ, четыре реконструкции интенсивности и вернуть метрики.

    Входные 1.png–4.png — амплитуды, uint8 L 128×128 в порядке расстояний.
    Метрики сравнивают их с амплитудами Френеля из сохранённого ДОЭ.
    При flat_reconstructions реконструкции сохраняются рядом с doe.png.
    """
    metrics, _ = _synthesize_quartet(
        input_directory,
        output_directory,
        model_path,
        seed=seed,
        flat_reconstructions=flat_reconstructions,
    )
    return metrics


def _synthesize_quartet(
    input_directory: str | Path,
    output_directory: str | Path,
    model_path: str | Path | None,
    *,
    seed: int,
    flat_reconstructions: bool = False,
) -> tuple[dict[str, Any], float]:
    """Запустить сеть на снимке входов и опубликовать метрики и длительность синтеза."""
    input_directory = Path(input_directory).resolve()
    output_directory = Path(output_directory).resolve()
    model_path = _resolve_model_path(model_path)
    _check_directories(input_directory, output_directory)
    expected_names = set(PLANE_PNG_NAMES)
    png_names = {path.name for path in input_directory.iterdir() if path.suffix.lower() == ".png"}
    if png_names != expected_names:
        raise ValueError(
            f"В {input_directory} нужны ровно 1.png, 2.png, 3.png, 4.png. "
            f"Отсутствуют: {sorted(expected_names - png_names)}; "
            f"лишние PNG: {sorted(png_names - expected_names)}."
        )
    result_names = FLAT_RESULT_NAMES if flat_reconstructions else RESULT_NAMES
    reconstruction_directory = Path() if flat_reconstructions else Path("reconstructions")
    _check_destination(output_directory, result_names)
    target_pixels, input_contents = _load_input_snapshot(input_directory, expected_names)
    amplitudes = normalize_pixels(target_pixels)
    black_planes = np.flatnonzero(np.max(amplitudes, axis=(-2, -1)) == 0) + 1
    if black_planes.size:
        raise ValueError(
            f"Полностью чёрные цели {black_planes.tolist()}: "
            "нормировка сети по среднему требует положительной яркости каждой плоскости."
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Не найден checkpoint: {model_path}. "
            "Запустите train.py или укажите путь к готовой модели через --model."
        )
    with tf.device("/CPU:0"):
        model, checkpoint_hash = _load_model_snapshot(model_path, seed)
        if not isinstance(model.network, LearnedUnrolled):
            raise ValueError("Пользовательский запуск требует checkpoint LearnedUnrolled.")
        target_intensities = tf.constant(np.square(amplitudes[None]), tf.float32)
        started_at = perf_counter()
        predicted_phase = model.predict_phase(target_intensities).numpy()
        synthesis_seconds = perf_counter() - started_at
        if predicted_phase.shape != (1, *const.IMAGE_SHAPE):
            raise ValueError("Сеть должна вернуть одну фазу 128×128.")
        phase_map = predicted_phase[0]
        if not np.isfinite(phase_map).all():
            raise ValueError("Фаза сети содержит NaN/Inf.")
        if np.any(np.abs(phase_map) > np.float32(np.pi)):
            raise ValueError("Фаза сети выходит за диапазон [−π, π].")
        evaluator = SavedDoeEvaluator()
        with _staged_publication(output_directory, result_names) as staging:
            save_phase(staging / "doe.png", phase_map)
            evaluation = evaluator.evaluate(
                staging / "doe.png",
                staging / reconstruction_directory,
                target_pixels,
                compare_amplitudes=True,
            )
            planes = []
            for plane in evaluation.planes:
                index = plane["plane"]
                pearson = plane["pearson"]
                planes.append(
                    {
                        "plane": index,
                        "target": f"inputs/{index}.png",
                        "reconstruction": (reconstruction_directory / f"{index}.png").as_posix(),
                        "distance_m": plane["distance_m"],
                        "pearson": pearson,
                        "ssim": plane["ssim"],
                    }
                )
                LOGGER.info(
                    "%d.png, z=%.1f мм: Pearson=%s, SSIM=%.6f",
                    index,
                    plane["distance_m"] * 1000,
                    f"{pearson:.6f}" if pearson is not None else "не определён",
                    plane["ssim"],
                )
            pearson_values = [plane["pearson"] for plane in planes if plane["pearson"] is not None]
            metrics = {
                "phase_source": "reloaded doe.png; 8-bit quantization included",
                "comparison": "target PNG amplitude vs propagated |K| before PNG quantization",
                "normalization": "each amplitude image divided by its own maximum; zeros kept",
                "reconstruction_png": "round(255 * |K|^2 / max(|K|^2)) per plane",
                "metric_scale": [0, 1],
                "ssim_variant": "global population statistics (ddof=0), no sliding window",
                "ssim_c1": 0.01**2,
                "ssim_c2": 0.03**2,
                "undefined_pearson": "null for constant target or reconstruction",
                "planes": planes,
                "mean_pearson": fmean(pearson_values) if pearson_values else None,
                "mean_ssim": fmean(plane["ssim"] for plane in planes),
                "defined_pearson_count": len(pearson_values),
            }
            _write_json(staging / "metrics.json", metrics)
            (staging / "inputs").mkdir()
            for name, image_bytes in input_contents.items():
                (staging / "inputs" / name).write_bytes(image_bytes)
            manifest = {
                "status": "complete",
                "input_directory": str(input_directory),
                "output_directory": str(output_directory),
                "checkpoint": {"path": str(model_path), "sha256": checkpoint_hash},
                "network": {
                    "architecture": "LearnedUnrolled",
                    "stages": model.network.stages,
                    "initial_phase_seed": model.network.initial_phase_seed,
                    "parameter_count": model.network.count_params(),
                    "runtime_weight_updates": 0,
                    "target_intensity": "(PNG amplitude / 255)^2",
                },
                "timing": {"synthesis_seconds": synthesis_seconds},
                "physics": {
                    "wavelength_m": const.WAVELENGTH_M,
                    "pixel_pitch_m": const.PIXEL_PITCH_M,
                    "distances_m": const.RECONSTRUCTION_DISTANCES_M,
                    "output_pixel_pitch_m": evaluator.output_pixel_pitch_m,
                    "illumination": "uniform A=1; H=A*exp(i*phase)",
                    "propagation": "centered single-FFT Fresnel per plane, full prefactor",
                    "phase_encoding": "clip(round(255*(phase+pi)/(2*pi)), 0, 255)",
                    "target_sampling": "each target uses its distance-dependent output grid",
                },
                "environment": {
                    "python": platform.python_version(),
                    "tensorflow": tf.__version__,
                    "numpy": np.__version__,
                    "pillow": PIL.__version__,
                    "device": "CPU",
                    "seed": seed,
                    "deterministic_ops": True,
                    "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
                    "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
                },
                "files_sha256": {
                    path.relative_to(staging).as_posix(): _sha256(path)
                    for path in sorted(staging.rglob("*"))
                    if path.is_file()
                },
            }
            _write_json(staging / "run.json", manifest)
    LOGGER.info(
        "Синтез занял %.6f с. ДОЭ и четыре реконструкции сохранены в %s",
        synthesis_seconds,
        output_directory,
    )
    return metrics, synthesis_seconds


def build_parser() -> argparse.ArgumentParser:
    """Создать CLI пользовательского синтеза без обучения на фотографиях."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Папка с комплектами PNG")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Папка результатов")
    parser.add_argument(
        "--model",
        type=Path,
        help="Checkpoint LearnedUnrolled (output/model.keras или weights/model.keras)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2, help="Число intra/inter-op потоков CPU")
    return parser


def _synthesize_input(args: argparse.Namespace) -> None:
    """Обработать непосредственные подпапки или одну четвёрку в указанной папке."""
    input_directory, output_directory = args.input.resolve(), args.output.resolve()
    _check_directories(input_directory, output_directory)
    entries = list(input_directory.iterdir())
    quartets = sorted(path for path in entries if path.is_dir())
    run_quartet = partial(
        _synthesize_quartet, model_path=_resolve_model_path(args.model), seed=args.seed
    )
    if not quartets:
        if not any(path.suffix.lower() == ".png" for path in entries):
            raise ValueError(f"В {input_directory} нет папок с комплектами 1.png–4.png.")
        run_quartet(input_directory, output_directory)
        return

    with _staged_publication(output_directory, (SUMMARY_NAME,)) as staging:
        results = []
        for quartet in quartets:
            destination = output_directory / quartet.name
            try:
                if quartet.name.casefold() in {SUMMARY_NAME, RESULT_LOCK_NAME}:
                    raise ValueError(f"Имя комплекта зарезервировано: {quartet.name}.")
                if destination.resolve() != destination:
                    raise ValueError(
                        f"Каталог результатов должен находиться по пути {destination}."
                    )
                LOGGER.info("Комплект %s", quartet.name)
                metrics, synthesis_seconds = run_quartet(
                    quartet, destination, flat_reconstructions=True
                )
            except (OSError, ValueError, tf.errors.OpError) as error:
                LOGGER.error("Комплект %s: %s", quartet.name, error)
                results.append({"name": quartet.name, "status": "failed", "error": str(error)})
            else:
                results.append(
                    {
                        "name": quartet.name,
                        "status": "complete",
                        "metrics": (Path(quartet.name) / "metrics.json").as_posix(),
                        "mean_pearson": metrics["mean_pearson"],
                        "mean_ssim": metrics["mean_ssim"],
                        "defined_pearson_count": metrics["defined_pearson_count"],
                        "synthesis_seconds": synthesis_seconds,
                    }
                )
        summary = _batch_summary(results)
        _write_json(staging / SUMMARY_NAME, summary)
    failed = summary["counts"]["failed"]
    if failed:
        raise ValueError(f"Не удалось обработать {failed} из {len(quartets)} комплектов.")


def _batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Усреднить метрики по успешным комплектам с равными весами."""
    completed = [result for result in results if result["status"] == "complete"]
    pearson_values = [
        result["mean_pearson"] for result in completed if result["mean_pearson"] is not None
    ]
    failed = len(results) - len(completed)
    status = "complete"
    if failed:
        status = "partial" if completed else "failed"
    return {
        "status": status,
        "counts": {"total": len(results), "succeeded": len(completed), "failed": failed},
        "mean_pearson": fmean(pearson_values) if pearson_values else None,
        "mean_ssim": fmean(result["mean_ssim"] for result in completed) if completed else None,
        "defined_pearson_quartet_count": len(pearson_values),
        "quartets": results,
    }


def main() -> None:
    """Проверить аргументы и выполнить синтез; ошибка возвращает код 2."""
    parser = build_parser()
    args = parser.parse_args()
    if args.threads <= 0 or args.seed < 0:
        parser.error("--threads должен быть положительным, --seed — неотрицательным.")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        _synthesize_input(args)
    except (OSError, ValueError, tf.errors.OpError) as error:
        LOGGER.error("Синтез не выполнен: %s", error)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
