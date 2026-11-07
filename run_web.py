"""Run in PyCharm; visit http://127.0.0.1:8000. One ASGI process, bounded threads."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
if __name__ == '__main__':
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    uvicorn.run('web_app:app',host=args.host,port=args.port,workers=1,limit_concurrency=64,timeout_keep_alive=10)
