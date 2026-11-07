"""Bounded, anonymous in-process jobs. Heavy work never runs on the ASGI loop."""
from __future__ import annotations
import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import io
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4
import warnings
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageOps
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pictures import load_phase_png, load_phase_txt
from simulator import simulate, export_results
from neural import experiment as engine
from neural.improve import Loss

log = logging.getLogger('doe.jobs')
router = APIRouter()
MAX_FILE = 4 * 1024 * 1024
MAX_BODY = 24 * 1024 * 1024
TERMINAL = {'ready', 'failed', 'cancelled'}
CHECKPOINT = Path(__file__).resolve().parent / 'neural' / 'model.npz'


class Cancelled(Exception): pass


@dataclass
class Job:
    id: str
    kind: str
    state: str = 'queued'
    progress: int = 0
    message: str = 'Ожидание свободного потока'
    revision: int = 0
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    artifacts: dict = field(default_factory=dict)
    metrics: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


class JobManager:
    def __init__(self, workers=2, capacity=24, ttl=3600):
        self.workers = workers
        self.capacity = capacity
        self.ttl = ttl
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.closed = False
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='doe-worker')

    def _prune(self):
        now = time.monotonic()
        for key, job in list(self.jobs.items()):
            if job.finished is not None and now - job.finished >= self.ttl:
                del self.jobs[key]

    def submit(self, kind, data, steps=600):
        with self.lock:
            self._prune()
            if self.closed: raise HTTPException(503, 'Сервер останавливается.')
            if len(self.jobs) >= self.capacity:
                completed = [j for j in self.jobs.values() if j.state in TERMINAL]
                if completed:
                    del self.jobs[min(completed, key=lambda j: j.finished).id]
                else:
                    raise HTTPException(503, 'Очередь заполнена. Повторите позднее.', headers={'Retry-After': '10'})
            job = Job(uuid4().hex, kind)
            self.jobs[job.id] = job
            self.pool.submit(self._run, job, data, steps)
            return self.snapshot(job.id)

    def update(self, job, **values):
        with self.lock:
            for key, value in values.items(): setattr(job, key, value)
            job.revision += 1

    def snapshot(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs.get(job_id)
            if job is None: raise HTTPException(404, 'Задача не найдена или срок хранения истёк.')
            payload = {'id': job.id, 'kind': job.kind, 'status': job.state,
                       'progress': job.progress, 'message': job.message, 'revision': job.revision,
                       'metrics': job.metrics, 'details': job.details,
                       'websocket_url': f'/ws/jobs/{job.id}', 'ttl_minutes': self.ttl // 60}
            if job.state == 'ready':
                payload['files'] = {name: f'/api/jobs/{job.id}/files/{name}' for name in job.artifacts}
            return payload

    def cancel_job(self, job_id):
        with self.lock:
            self.snapshot(job_id)
            job = self.jobs[job_id]
            if job.state not in TERMINAL:
                job.cancel.set()
                self.update(job, message='Остановка расчёта…')
            return self.snapshot(job_id)

    def _check_cancel(self, job):
        if job.cancel.is_set(): raise Cancelled()

    def _run(self, job, data, steps):
        try:
            self._check_cancel(job)
            self.update(job, state='running', progress=2, message='Подготовка изображений')
            artifacts, metrics, details = self._synthesis(job, data, steps) if job.kind == 'synthesis' else self._phase(job, data)
            self._check_cancel(job)
            self.update(job, artifacts=artifacts, metrics=metrics, details=details,
                        state='ready', progress=100, message='Расчёт завершён', finished=time.monotonic())
        except Cancelled:
            self.update(job, state='cancelled', message='Расчёт отменён', finished=time.monotonic())
        except Exception:
            log.exception('Job %s failed', job.id)
            self.update(job, state='failed', message='Не удалось выполнить расчёт. Попробуйте другой комплект изображений.', finished=time.monotonic())

    def _phase(self, job, phase):
        self.update(job, progress=20, message='Распространение Френеля: четыре плоскости')
        self._check_cancel(job)
        result = simulate(phase, source='web:uploaded-phase')
        self.update(job, progress=65, message='Экспорт фазовой маски и восстановлений')
        with tempfile.TemporaryDirectory(prefix='doe-phase-') as directory:
            dest = Path(directory)/'result'
            export_results(result, dest)
            quantized = load_phase_png(dest/'doe.png')
            np.savetxt(dest/'phase_matrix.txt', quantized, fmt='%.9f', delimiter=' ')
            artifacts = {'phase.png': (dest/'doe.png').read_bytes(),
                         'phase_matrix.txt': (dest/'phase_matrix.txt').read_bytes()}
            for i in range(4): artifacts[f'reconstruction_{i+1}.png'] = (dest/'quantized'/f'{i+1}.png').read_bytes()
            artifacts['result.zip'] = zip_directory(dest)
        return artifacts, [], {'normalization': 'PNG: max intensity independently per plane; raw NPY in archive', 'method': 'Existing simulator.simulate; single-FFT Fresnel'}

    def _synthesis(self, job, original, steps):
        model = engine.MLP.load(CHECKPOINT)  # Private mutable weights per job.
        x = engine.features(original)
        start = model.forward(x).copy()
        loss = Loss(original, (.3, 1, .5, .6))
        opt = engine.Adam([model.w1, model.w2], [.00005, .015])
        best = float('inf'); chosen = start.copy()
        # Fit this user's images; the bundled one-example model is only a warm start.
        for step in range(steps+1):
            self._check_cancel(job)
            phase = model.forward(x)
            value, gradient = loss.evaluate(phase)
            if value < best: best = value; chosen = phase.copy()
            if step % 25 == 0 or step == steps:
                self.update(job, progress=5+int(85*step/max(steps, 1)),
                            message=f'Настройка общей фазы: {step} / {steps}',
                            details={'step': step, 'steps': steps, 'loss': value})
            if step < steps:
                decay = .1 + .9*.5*(1+math.cos(math.pi*step/max(steps, 1)))
                opt.step(model.backward(gradient), decay)
        self.update(job, progress=92, message='Квантование фазы и независимая проверка восстановления')
        self._check_cancel(job)
        with tempfile.TemporaryDirectory(prefix='doe-synthesis-') as directory:
            dest = Path(directory)
            for i, a in enumerate(original): Image.fromarray(np.rint(a*255).astype(np.uint8)).save(dest/f'input_{i+1}.png')
            # Export and reopen actual 8-bit phase before computing outputs/metrics.
            engine.save_result(dest, chosen, original, engine.chirps())
            decoded = load_phase_png(dest/'phase_mask.png')
            physical = simulate(decoded, source='web:neural-synthesis')
            # Use the CURRENT project's physical operator, normalized only for display.
            # Its prefactor differs from the normalized training operator by a scalar per plane.
            raw = physical.intensities / physical.intensities.mean(axis=(1,2), keepdims=True)
            raw *= original.mean(axis=(1,2), keepdims=True)
            np.save(dest/'physical_intensities.npy', physical.intensities)
            np.save(dest/'brightness_unclipped.npy', raw)
            pixels = np.rint(np.clip(raw, 0, 1)*255).astype(np.uint8)
            metrics = engine.metrics(original, pixels/255.)
            for i, pixels_i in enumerate(pixels): Image.fromarray(pixels_i).save(dest/f'reconstruction_{i+1}.png')
            metadata = {'method': 'One-example MLP fitting through single-FFT Fresnel', 'steps': steps,
                        'metrics': metrics, 'metrics_unclipped': engine.metrics(original, raw),
                        'names': ['image_1','image_2','image_3','image_4'],
                        'normalization': 'Mean intensity matched to each target; no fitted gain or smoothing',
                        'limitation': 'Quality and crosstalk suppression are not guaranteed for new images.',
                        'wavelength_nm': 532, 'slm_pitch_um': 10, 'z_mm': [24.1,25.3,26.5,27.7]}
            engine.write_json(dest/'metrics.json', metadata)
            artifacts = {'phase.png': (dest/'phase_mask.png').read_bytes(),
                         'phase_matrix.txt': (dest/'phase_rad.txt').read_bytes(),
                         'metrics.json': (dest/'metrics.json').read_bytes()}
            for i in range(4):
                artifacts[f'input_{i+1}.png'] = (dest/f'input_{i+1}.png').read_bytes()
                artifacts[f'reconstruction_{i+1}.png'] = (dest/f'reconstruction_{i+1}.png').read_bytes()
            artifacts['result.zip'] = zip_directory(dest)
        return artifacts, metrics, {'steps': steps, 'loss': best, 'normalization': metadata['normalization']}

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                if job.state not in TERMINAL: job.cancel.set()
        self.pool.shutdown(wait=True, cancel_futures=False)


def zip_directory(path):
    stream = io.BytesIO()
    with ZipFile(stream, 'w', ZIP_DEFLATED) as archive:
        for child in path.iterdir():
            if child.is_file(): archive.write(child, child.name)
    return stream.getvalue()


@asynccontextmanager
async def job_lifespan(app):
    workers = max(1, min(int(os.getenv('DOE_WORKERS', '2')), 8))
    app.state.jobs = JobManager(workers=workers)
    try: yield
    finally: await asyncio.to_thread(app.state.jobs.close)


def check_origin(request):
    origin = request.headers.get('origin')
    # Same-origin browser frontend. Non-browser local API clients may omit Origin.
    if origin:
        from urllib.parse import urlsplit
        if urlsplit(origin).netloc != request.headers.get('host'):
            raise HTTPException(403, 'Недопустимый источник запроса.')


async def read_payload(request):
    check_origin(request)
    content = bytearray()
    async for part in request.stream():
        if len(content) + len(part) > MAX_BODY: raise HTTPException(413, 'Слишком большой запрос.')
        content.extend(part)
    try:
        obj = json.loads(content)
        if not isinstance(obj, dict): raise ValueError()
        return obj
    except (ValueError, UnicodeError): raise HTTPException(422, 'Ожидается JSON-объект.') from None


def decode_file(item):
    if not isinstance(item, dict) or not isinstance(item.get('data'), str): raise ValueError('Некорректный файл.')
    if len(item['data']) > (MAX_FILE*4//3+8): raise ValueError('Размер каждого файла — не более 4 МиБ.')
    try: data = base64.b64decode(item['data'], validate=True)
    except (ValueError, binascii.Error): raise ValueError('Неверная кодировка файла.') from None
    if not data or len(data) > MAX_FILE: raise ValueError('Пустой или слишком большой файл.')
    return data


def prepare_targets(items):
    if not isinstance(items, list) or len(items) != 4: raise ValueError('Нужны ровно четыре изображения.')
    output = []
    for item in items:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(decode_file(item))) as im:
                if im.width * im.height > 16_000_000: raise ValueError('Изображение слишком большое: максимум 16 Мп.')
                if im.format not in {'PNG','JPEG','WEBP','BMP','TIFF'}: raise ValueError('Поддерживаются PNG, JPEG, WEBP, BMP и TIFF.')
                rgba = ImageOps.exif_transpose(im).convert('RGBA')
                image = Image.alpha_composite(Image.new('RGBA', rgba.size, (0,0,0,255)), rgba).convert('L')
                image = ImageOps.pad(image,(128,128), method=Image.Resampling.LANCZOS, color=0)
                a = np.asarray(image, dtype=np.float64)/255.
                if np.std(a) < 1e-8: raise ValueError('Выберите изображения с ненулевым контрастом.')
                output.append(a)
    return np.stack(output)


@router.get('/api/health')
def health(request: Request):
    manager = request.app.state.jobs
    with manager.lock:
        return {'status':'ok', 'workers':manager.workers,
                'running':sum(j.state=='running' for j in manager.jobs.values()),
                'queued':sum(j.state=='queued' for j in manager.jobs.values()),
                'registration':False, 'ttl_minutes':60}


@router.post('/api/jobs/synthesis', status_code=202)
async def create_synthesis(request: Request):
    obj = await read_payload(request)
    steps = obj.get('steps', 600)
    if type(steps) is not int or not 0 <= steps <= 2000: raise HTTPException(422, 'Число шагов должно быть от 0 до 2000.')
    try: images = await asyncio.to_thread(prepare_targets, obj.get('images'))
    except (ValueError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(422, 'Проверьте четыре изображения: PNG/JPEG/WEBP/BMP/TIFF, до 4 МиБ и 16 Мп каждое, ненулевой контраст.') from None
    return request.app.state.jobs.submit('synthesis', images, steps)


@router.post('/api/jobs/phase', status_code=202)
async def create_phase(request: Request):
    obj = await read_payload(request)
    try:
        data = decode_file(obj)
        if obj.get('kind') == 'png':
            phase = await asyncio.to_thread(load_phase_png, io.BytesIO(data))
        elif obj.get('kind') == 'txt':
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'phase.txt'; path.write_bytes(data)
                phase = await asyncio.to_thread(load_phase_txt, path)
        else: raise ValueError('Неверный формат.')
    except (ValueError, OSError, UnicodeError): raise HTTPException(422, 'Ожидается фазовый PNG или TXT 128×128 в радианах.') from None
    return request.app.state.jobs.submit('phase', phase)


@router.get('/api/jobs/{job_id}')
def job_status(job_id: str, request: Request):
    return request.app.state.jobs.snapshot(job_id)


@router.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str, request: Request):
    check_origin(request)
    return request.app.state.jobs.cancel_job(job_id)


@router.get('/api/jobs/{job_id}/files/{filename}')
def job_file(job_id: str, filename: str, request: Request):
    manager = request.app.state.jobs
    with manager.lock:
        status = manager.snapshot(job_id)
        if status['status'] != 'ready': raise HTTPException(409, 'Результат ещё не готов.')
        data = manager.jobs[job_id].artifacts.get(filename)
    if data is None: raise HTTPException(404, 'Файл не найден.')
    content_type = 'image/png' if filename.endswith('.png') else 'application/zip' if filename.endswith('.zip') else 'application/json' if filename.endswith('.json') else 'text/plain; charset=utf-8'
    headers={'Cache-Control':'no-store'}
    if not filename.endswith('.png'): headers['Content-Disposition']=f'attachment; filename="{filename}"'
    return Response(data, media_type=content_type, headers=headers)


@router.websocket('/ws/jobs/{job_id}')
async def job_socket(websocket: WebSocket, job_id: str):
    try: check_origin(websocket)
    except HTTPException:
        await websocket.close(code=1008); return
    manager = websocket.app.state.jobs
    try: manager.snapshot(job_id)
    except HTTPException:
        await websocket.close(code=1008); return
    await websocket.accept()
    revision = -1
    try:
        while True:
            state = manager.snapshot(job_id)
            if state['revision'] != revision:
                await websocket.send_json(state)
                revision = state['revision']
            if state['status'] in TERMINAL:
                await websocket.close(code=1000); return
            try:
                packet = await asyncio.wait_for(websocket.receive(), timeout=.2)
                if packet['type']=='websocket.disconnect': return
            except asyncio.TimeoutError: pass
    except (WebSocketDisconnect, HTTPException): return
