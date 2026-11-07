"""FastAPI-оболочка для существующего симулятора фазового ДОЭ.

Запуск: uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload

Расчёты проводятся исключительно функцией simulator.simulate; интерфейс
не меняет физическую модель, входной масштаб фазы или нормировку интенсивности.

Демонстрационный сервер хранит результаты в памяти одного процесса не более
одного часа. Для высоконагруженного многопроцессного хостинга нужны внешние
хранилище задач, ограничения доступа и очередь вычислений.
"""

from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import io
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from web_jobs import router as jobs_router, job_lifespan
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, StrictInt
from PIL import Image

from const import (
    DEFAULT_AMPLITUDE,
    IMAGE_SIZE,
    PIXEL_PITCH_M,
    RECONSTRUCTION_DISTANCES_M,
    WAVELENGTH_M,
)
from mathlib import output_pixel_pitch
from pictures import OutputPictures, load_phase_png, load_phase_txt
from simulator import (
    SimulationResult,
    export_results,
    generate_random_phase,
    simulate,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_BYTES = 1_048_576
MAX_JOBS = 30
JOB_TTL_SECONDS = 60 * 60

app = FastAPI(
    lifespan=job_lifespan,
    title="Fresnel DOE Simulator",
    description="Одна фазовая маска 128×128 → четыре плоскости дифракции Френеля.",
    version="1.0.0",
)
app.include_router(jobs_router)


class RandomPhaseRequest(BaseModel):
    seed: StrictInt | None = Field(default=None, ge=0, le=2**53 - 1)


@dataclass
class Job:
    phase: np.ndarray
    seed: int | None
    source: str
    created_at: float = field(default_factory=time.monotonic)
    result: SimulationResult | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.RLock()


def _cleanup_jobs() -> None:
    """Удалить просроченные задачи; ограничить память одного процесса."""
    now = time.monotonic()
    with _jobs_lock:
        for key, job in list(_jobs.items()):
            if now - job.created_at >= JOB_TTL_SECONDS and not job.lock.locked():
                _jobs.pop(key, None)
        while len(_jobs) >= MAX_JOBS:
            candidates = [(key, item) for key, item in _jobs.items()
                          if not item.lock.locked()]
            if not candidates:
                raise HTTPException(503, "Сервер занят. Повторите запрос позже.")
            oldest_key = min(candidates, key=lambda pair: pair[1].created_at)[0]
            _jobs.pop(oldest_key, None)


def _create_job(phase: np.ndarray, seed: int | None, source: str) -> dict:
    _cleanup_jobs()
    job_id = uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = Job(phase=phase, seed=seed, source=source)
    return {
        "id": job_id,
        "seed": seed,
        "source": source,
        "phase_url": f"/api/phases/{job_id}/preview",
        "status": "phase_ready",
    }


def _get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or time.monotonic() - job.created_at >= JOB_TTL_SECONDS:
            _jobs.pop(job_id, None)
            raise HTTPException(404, "Расчёт не найден или срок его хранения истёк.")
        return job


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _phase_image(phase: np.ndarray) -> Image.Image:
    """Та же фиксированная визуальная шкала фазы, что и в Tk GUI."""
    stops = np.array([
        [37, 48, 118], [39, 127, 158], [53, 184, 153],
        [179, 218, 83], [251, 231, 100],
    ], dtype=np.float64)
    values = np.clip((phase + np.pi) / (2 * np.pi), 0, 1)
    positions = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(values, positions, stops[:, channel])
                    for channel in range(3)], axis=-1)
    return Image.fromarray(np.rint(rgb).astype(np.uint8))


def _require_result(job_id: str) -> SimulationResult:
    job = _get_job(job_id)
    if job.result is None:
        raise HTTPException(409, "Сначала нажмите «Рассчитать дифракцию».")
    return job.result


def _result_payload(job_id: str, result: SimulationResult) -> dict:
    screens = []
    for index, distance in enumerate(RECONSTRUCTION_DISTANCES_M, start=1):
        pitch = output_pixel_pitch(distance)
        screens.append({
            "index": index,
            "distance_mm": distance * 1e3,
            "pixel_pitch_um": pitch * 1e6,
            "area_mm": IMAGE_SIZE * pitch * 1e3,
            "peak_intensity": float(result.intensities[index - 1].max()),
            "image_url": f"/api/simulations/{job_id}/screens/{index}",
        })
    return {"id": job_id, "status": "ready", "screens": screens,
            "download_url": f"/api/simulations/{job_id}/download"}


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/styles.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")


@app.get("/api/config")
def config() -> dict:
    return {
        "image_size": IMAGE_SIZE,
        "wavelength_nm": WAVELENGTH_M * 1e9,
        "slm_pixel_pitch_um": PIXEL_PITCH_M * 1e6,
        "amplitude": DEFAULT_AMPLITUDE,
        "distances_mm": [distance * 1e3 for distance in RECONSTRUCTION_DISTANCES_M],
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "ttl_minutes": JOB_TTL_SECONDS // 60,
    }


@app.post("/api/phases/random", status_code=201)
def random_phase(payload: RandomPhaseRequest = Body(default=RandomPhaseRequest())) -> dict:
    phase = generate_random_phase(payload.seed)
    return _create_job(phase, payload.seed, "random")


@app.post("/api/phases/upload", status_code=201)
async def upload_phase(
    request: Request,
    kind: Literal["png", "txt"] = Query(...),
    filename: str = Query(default="uploaded"),
) -> dict:
    """Отправьте сырые байты PNG/TXT в теле запроса (не multipart)."""
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Файл слишком большой (максимум 1 МиБ).")
        except ValueError:
            raise HTTPException(400, "Некорректный Content-Length.") from None

    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Файл слишком большой (максимум 1 МиБ).")
    if not data:
        raise HTTPException(400, "Файл пуст.")

    try:
        if kind == "png":
            phase = load_phase_png(io.BytesIO(data))
        else:
            # Используем оригинальный строгий парсер TXT без дублирования кода.
            with tempfile.TemporaryDirectory(prefix="fresnel-upload-") as folder:
                path = Path(folder) / "phase.txt"
                path.write_bytes(data)
                phase = load_phase_txt(path)
    except (ValueError, OSError, UnicodeError) as exc:
        raise HTTPException(422, str(exc)) from exc

    clean_name = re.sub(r"[\x00-\x1f\x7f]", "", filename.replace("\\", "/").split("/")[-1])[:100]
    return _create_job(phase, None, f"upload:{clean_name}")


@app.get("/api/phases/{job_id}/preview")
def preview_phase(job_id: str) -> Response:
    phase = _get_job(job_id).phase
    return Response(_png_bytes(_phase_image(phase)), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/simulations/{job_id}")
def calculate(job_id: str) -> dict:
    job = _get_job(job_id)
    # Один и тот же job может запрашиваться повторно; вычисляем только один раз.
    with job.lock:
        if job.result is None:
            job.result = simulate(job.phase, seed=job.seed, source=job.source)
        return _result_payload(job_id, job.result)


@app.get("/api/simulations/{job_id}")
def get_result(job_id: str) -> dict:
    return _result_payload(job_id, _require_result(job_id))


@app.get("/api/simulations/{job_id}/screens/{index}")
def screen_image(job_id: str, index: int) -> Response:
    if not 1 <= index <= len(RECONSTRUCTION_DISTANCES_M):
        raise HTTPException(404, "Экран должен иметь номер от 1 до 4.")
    result = _require_result(job_id)
    pixels = OutputPictures(Path("."), result.intensities).to_uint8()
    image = Image.fromarray(pixels[index - 1])
    return Response(_png_bytes(image), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/simulations/{job_id}/download")
def download(job_id: str) -> Response:
    result = _require_result(job_id)
    # Исходный экспорт уже содержит квантованный doe.png, NPY и метаданные.
    # Упаковываем существующий формат без потери точности.
    with tempfile.TemporaryDirectory(prefix="fresnel-export-") as folder:
        target = Path(folder) / f"fresnel_{job_id[:8]}"
        export_results(result, target)
        memory = io.BytesIO()
        with ZipFile(memory, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(target.parent))
    filename = f"fresnel_{job_id[:8]}.zip"
    return Response(
        memory.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('web_app:app', host='127.0.0.1', port=8000, workers=1, limit_concurrency=64)
