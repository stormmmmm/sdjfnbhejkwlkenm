"""Потоковая загрузка четвёрок из <split>/<имя>/1.png–4.png."""

from copy import copy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from utils.const import IMAGE_SHAPE, INPUT_IMAGE_COUNT
from utils.pictures import load_grayscale_png, normalize_pixels


class ManifestEntry(TypedDict):
    """Имя четвёрки и SHA-256 исходных файлов по имени PNG."""

    sample: str
    sha256: dict[str, str]


@dataclass(frozen=True)
class _SelectedQuartet:
    paths: tuple[Path, ...]
    hashes: tuple[str, ...]


def discover_quartets(root: str | Path) -> tuple[tuple[Path, ...], ...]:
    """Найти любое число четвёрок и вернуть пути в порядке плоскостей.

    Каждая подпапка — один пример с 1.png, 2.png, 3.png, 4.png.
    Дополнительные файлы игнорируются. Числовые имена сортируются численно,
    остальные — по имени после числовых;
    содержимое PNG читается позже, при обходе набора.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Папка изображений не найдена: {root}")
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: (
            (0, int(path.name), path.name) if path.name.isdecimal() else (1, 0, path.name)
        ),
    )

    filenames = tuple(f"{index}.png" for index in range(1, INPUT_IMAGE_COUNT + 1))
    quartets = []
    for directory in directories:
        paths = tuple(directory / name for name in filenames)
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"В папке {directory} отсутствуют файлы: {', '.join(missing)}.")
        quartets.append(paths)
    return tuple(quartets)


class TrainingPictures:
    """Набор четвёрок любого размера для train, validation или test.

    При выборе фиксируются пути и хеши PNG. Декодирование происходит
    потоково при обходе и использует те же байты, чей хеш проверен.
    """

    def __init__(self, root: str | Path, count: int | None = None) -> None:
        self.root = Path(root)
        quartets = discover_quartets(self.root)
        if count is not None:
            self._validate_count(count, len(quartets))
            quartets = quartets[:count]
        self._selection = tuple(
            _SelectedQuartet(paths, tuple(sha256(path.read_bytes()).hexdigest() for path in paths))
            for paths in quartets
        )

    def __len__(self) -> int:
        """Вернуть количество обучающих четвёрок."""
        return len(self._selection)

    @staticmethod
    def _validate_count(count: int, available: int) -> None:
        if not 1 <= count <= available:
            raise ValueError(f"Число примеров должно быть от 1 до {available}.")

    @property
    def quartets(self) -> tuple[tuple[Path, ...], ...]:
        """Вернуть зафиксированные пути четвёрок в порядке плоскостей."""
        return tuple(quartet.paths for quartet in self._selection)

    def prefix(self, count: int) -> "TrainingPictures":
        """Взять первые count четвёрок с исходными хешами, без обращения к диску."""
        self._validate_count(count, len(self))
        subset = copy(self)
        subset._selection = self._selection[:count]
        return subset

    def manifest(self) -> list[ManifestEntry]:
        """Вернуть имена и контрольные суммы на момент выбора четвёрок."""
        return [
            {
                "sample": quartet.paths[0].parent.name,
                "sha256": {
                    path.name: digest
                    for path, digest in zip(quartet.paths, quartet.hashes, strict=True)
                },
            }
            for quartet in self._selection
        ]

    def read_quartet(self, index: int) -> NDArray[np.uint8]:
        """Проверить SHA-256 и декодировать четвёрку uint8 формы 4×128×128."""
        quartet = self._selection[index]
        images = []
        for path, digest in zip(quartet.paths, quartet.hashes, strict=True):
            contents = path.read_bytes()
            if sha256(contents).hexdigest() != digest:
                raise ValueError(f"PNG изменился после выбора набора: {path} (SHA-256).")
            images.append(load_grayscale_png(path, contents=contents))
        return np.stack(images)

    def _read_normalized_quartet(self, index: np.int64) -> NDArray[np.float32]:
        return normalize_pixels(self.read_quartet(int(index)))

    def as_dataset(
        self, batch_size: int = 8, seed: int = 42, shuffle: bool = True
    ) -> tf.data.Dataset:
        """Вернуть конечный Dataset тензоров float32 B×4×128×128 в [0, 1].

        Перемешиваются индексы целых четвёрок; порядок плоскостей сохраняется.
        При повторном обходе меняется порядок четвёрок, воспроизводимый по seed.
        Последний неполный мини-пакет сохраняется. Буфер содержит один пакет.
        """
        if batch_size <= 0:
            raise ValueError("Размер мини-пакета должен быть положительным.")
        dataset = tf.data.Dataset.range(len(self))
        if shuffle and len(self):
            dataset = dataset.shuffle(len(self), seed=seed, reshuffle_each_iteration=True)

        def load_quartet(index: tf.Tensor) -> tf.Tensor:
            images = tf.numpy_function(self._read_normalized_quartet, [index], tf.float32)
            return tf.ensure_shape(images, (INPUT_IMAGE_COUNT, *IMAGE_SHAPE))

        dataset = dataset.map(load_quartet, num_parallel_calls=1, deterministic=True)
        return dataset.batch(batch_size, drop_remainder=False).prefetch(1)
