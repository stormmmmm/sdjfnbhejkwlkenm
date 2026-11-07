import base64
import io
import threading
import time
import unittest
from zipfile import ZipFile
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.testclient import TestClient
from web_app import app
from web_jobs import JobManager
from neural import experiment as engine
from simulator import simulate


def encoded(image):
    stream=io.BytesIO(); image.save(stream,format='PNG')
    return {'name':'input.png','data':base64.b64encode(stream.getvalue()).decode()}


class WebTests(unittest.TestCase):
    def setUp(self):
        self.context=TestClient(app); self.client=self.context.__enter__()
    def tearDown(self): self.context.__exit__(None,None,None)
    def await_job(self,job):
        with self.client.websocket_connect(job['websocket_url']) as ws:
            while True:
                data=ws.receive_json()
                if data['status'] in {'ready','failed','cancelled'}: return data
    def test_site_and_health(self):
        for path in ['/','/styles.css','/app.js']:
            self.assertEqual(self.client.get(path).status_code,200)
        data=self.client.get('/api/health').json()
        self.assertFalse(data['registration']); self.assertEqual(data['workers'],2)
    def test_photo_job_websocket_metrics_and_txt(self):
        rng=np.random.default_rng(12)
        images=[encoded(Image.fromarray(rng.integers(0,256,(128,128),dtype=np.uint8))) for _ in range(4)]
        response=self.client.post('/api/jobs/synthesis',json={'images':images,'steps':2})
        self.assertEqual(response.status_code,202,response.text)
        data=self.await_job(response.json())
        self.assertEqual(data['status'],'ready',data)
        self.assertEqual(data['progress'],100); self.assertEqual(len(data['metrics']),4)
        txt=self.client.get(data['files']['phase_matrix.txt']).text
        phase=np.loadtxt(io.StringIO(txt)); self.assertEqual(phase.shape,(128,128))
        self.assertTrue(np.isfinite(phase).all())
        png=self.client.get(data['files']['phase.png']).content
        with Image.open(io.BytesIO(png)) as im:
            self.assertEqual(im.mode,'L'); self.assertEqual(im.size,(128,128))
            decoded=engine.decode_phase(np.asarray(im))
        np.testing.assert_allclose(phase,decoded,atol=5.1e-10)
        # Agreement of normalized neural operator with the existing physical simulator.
        physical=simulate(decoded).intensities
        np.testing.assert_allclose(physical/physical.mean(axis=(1,2),keepdims=True),np.abs(engine.propagate(decoded,engine.chirps()))**2,rtol=1e-11,atol=1e-11)
        archive=self.client.get(data['files']['result.zip']).content
        with ZipFile(io.BytesIO(archive)) as z:
            self.assertIn('phase_rad.txt',z.namelist()); self.assertIn('metrics.json',z.namelist())
            self.assertEqual(len([n for n in z.namelist() if n.startswith('input_')]),4)
    def test_phase_job_and_no_fake_metrics(self):
        response=self.client.post('/api/jobs/phase',json={**encoded(Image.new('L',(128,128),128)),'kind':'png'})
        self.assertEqual(response.status_code,202)
        data=self.await_job(response.json())
        self.assertEqual(data['status'],'ready'); self.assertEqual(data['metrics'],[])
        self.assertEqual(self.client.get(data['files']['phase_matrix.txt']).status_code,200)
    def test_invalid_files_steps_and_origins(self):
        for payload in [{'images':[]},{'images':[{}]*4},{'images':[],'steps':True},{'images':[],'steps':2001}]:
            self.assertEqual(self.client.post('/api/jobs/synthesis',json=payload).status_code,422)
        response=self.client.post('/api/jobs/phase',json={'kind':'png','data':'NOT BASE64'})
        self.assertEqual(response.status_code,422)
        response=self.client.post('/api/jobs/synthesis',json={},headers={'Origin':'https://foreign.example'})
        self.assertEqual(response.status_code,403)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code,404)
    def test_old_api_preserved(self):
        response=self.client.post('/api/phases/random',json={'seed':32})
        self.assertEqual(response.status_code,201)
        self.assertEqual(self.client.get(response.json()['phase_url']).status_code,200)


class PoolTests(unittest.TestCase):
    def test_two_workers_run_simultaneously_with_distinct_jobs(self):
        manager=JobManager(workers=2,capacity=2)
        barrier=threading.Barrier(3); release=threading.Event(); seen=[]
        def work(job,data):
            seen.append((job.id,threading.current_thread().name,data))
            barrier.wait(timeout=3); release.wait(timeout=3)
            return {'result.zip':bytes([data])},[],{}
        manager._phase=work
        try:
            a=manager.submit('phase',1); b=manager.submit('phase',2)
            barrier.wait(timeout=3)
            self.assertEqual(len({thread for _,thread,_ in seen}),2)
            self.assertNotEqual(a['id'],b['id'])
            with self.assertRaises(HTTPException) as cm: manager.submit('phase',3)
            self.assertEqual(cm.exception.status_code,503)
            release.set()
            deadline=time.monotonic()+3
            while time.monotonic()<deadline and any(j.state not in {'ready','failed'} for j in manager.jobs.values()): time.sleep(.01)
            self.assertTrue(all(j.state=='ready' for j in manager.jobs.values()))
        finally: release.set(); manager.close()
        self.assertEqual(manager.jobs[a['id']].artifacts['result.zip'],b'\x01')
        self.assertEqual(manager.jobs[b['id']].artifacts['result.zip'],b'\x02')
    def test_queued_cancellation_and_ttl(self):
        manager=JobManager(workers=1,ttl=.01)
        entered=threading.Event(); release=threading.Event(); performed=[]
        def work(job,data):
            performed.append(data); entered.set(); release.wait(timeout=3)
            return {},[],{}
        manager._phase=work
        first=manager.submit('phase',1); self.assertTrue(entered.wait(3))
        second=manager.submit('phase',2); manager.cancel_job(second['id'])
        release.set(); manager.close()
        self.assertEqual(performed,[1]); self.assertEqual(manager.jobs[second['id']].state,'cancelled')
        time.sleep(.02)
        with self.assertRaises(HTTPException) as cm: manager.snapshot(first['id'])
        self.assertEqual(cm.exception.status_code,404)


if __name__=='__main__': unittest.main()
