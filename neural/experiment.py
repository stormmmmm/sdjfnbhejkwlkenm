"""One-example, physics-supervised MLP experiment. NumPy + Pillow only.

This is an overfit/feasibility test, not a generalizing hologram predictor.
Run: python experiment.py train --steps 1800 --seeds 3
     python experiment.py check
     python experiment.py infer --checkpoint results/model.npz --inputs inputs/01_cat.png inputs/02_pear.png inputs/03_mug.png inputs/04_lighthouse.png --out inference
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
N = 128
WL = 532e-9
DX = 10e-6
ZS = np.array([24.1, 25.3, 26.5, 27.7]) * 1e-3
NAMES = ['image_1', 'image_2', 'image_3', 'image_4']
EPS = 1e-10


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fft(a):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))


def ifft(a):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))


def chirps(n=N, zs=ZS):
    q = (np.arange(n) - n//2) * DX
    r2 = q[:,None]**2 + q[None,:]**2
    return np.exp(1j*np.pi*r2[None,:,:]/(WL*np.asarray(zs)[:,None,None]))


def propagate(phase, c):
    # Single-FFT Fresnel: global prefactor and output chirp cancel from
    # normalized intensity. Native output pitch = lambda*z/(N*DX).
    return fft(c * np.exp(1j*phase))


def objective(phase, target, c, gradient=True):
    e = np.exp(1j*phase)
    u = fft(c*e)
    intensity = np.abs(u)**2
    delta = np.sqrt(intensity+EPS)-np.sqrt(target+EPS)
    loss = float(np.mean(delta**2))
    if not gradient:
        return loss
    di = delta / (target.size*np.sqrt(intensity+EPS))
    back = np.sum(np.conj(c)*ifft(di*u),axis=0)
    g = 2*np.imag(np.conj(e)*back)
    return loss, g


def load_images(paths):
    arrays=[]
    for path in paths:
        with Image.open(path) as im:
            data=Path(path).read_bytes()
            if im.size!=(N,N) or im.mode!='L' or data[24]!=8 or data[25]!=0:
                raise ValueError(f'{path}: expected 128x128 grayscale PNG, 8 bit, color type 0')
            arrays.append(np.asarray(im,dtype=np.float64)/255.)
    a=np.stack(arrays)
    if len(a)!=4 or np.any(a.mean(axis=(1,2))<=0):
        raise ValueError('Need exactly four nonblack images')
    return a


def features(images):
    # Fixed average pooling: 4x128x128 -> 4x16x16 -> 1024.
    return (2*images.reshape(4,16,8,16,8).mean(axis=(2,4))-1).ravel()


class MLP:
    """1024 -> 32 tanh -> 16384 raw phases, no output bias/template."""
    def __init__(self,seed=0):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(size=(32,1024))
        self.w2=rng.normal(scale=3.,size=(N*N,32))

    def forward(self,x):
        self.x=x
        self.h=np.tanh(self.w1@x/np.sqrt(1024))
        return (self.w2@self.h/np.sqrt(32)).reshape(N,N)

    def backward(self,g):
        gout=g.ravel()/np.sqrt(32)
        gw2=np.outer(gout,self.h)
        gh=(self.w2.T@gout)*(1-self.h**2)
        gw1=np.outer(gh,self.x)/np.sqrt(1024)
        return gw1,gw2

    def save(self,path):
        np.savez_compressed(path,w1=self.w1,w2=self.w2)

    @classmethod
    def load(cls,path):
        obj=cls(0)
        with np.load(path,allow_pickle=False) as d:
            obj.w1=d['w1'].copy(); obj.w2=d['w2'].copy()
        return obj


class Adam:
    def __init__(self,arrays,rates):
        self.arrays=arrays; self.rates=rates
        self.m=[np.zeros_like(a) for a in arrays]
        self.v=[np.zeros_like(a) for a in arrays]; self.t=0

    def step(self,grads,factor=1.):
        self.t+=1
        for a,g,m,v,lr in zip(self.arrays,grads,self.m,self.v,self.rates):
            m*=.9; m+=.1*g
            v*=.999; v+=.001*g*g
            a-=lr*factor*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)


def pearson(a,b):
    x=a.ravel()-a.mean(); y=b.ravel()-b.mean()
    denom=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denom) if denom>1e-14 else None


def gaussian_valid(a):
    q=np.arange(-5,6,dtype=float)
    kernel=np.exp(-q*q/(2*1.5**2)); kernel/=kernel.sum()
    horizontal=np.einsum('ijk,k->ij',np.lib.stride_tricks.sliding_window_view(a,11,axis=1),kernel)
    return np.einsum('ijk,k->ij',np.lib.stride_tricks.sliding_window_view(horizontal,11,axis=0),kernel)


def ssim(a,b):
    # Local SSIM, gaussian 11x11 sigma=1.5, population covariance,
    # data_range=1, valid interior (no boundary padding).
    ma=gaussian_valid(a); mb=gaussian_valid(b)
    va=np.maximum(gaussian_valid(a*a)-ma*ma,0)
    vb=np.maximum(gaussian_valid(b*b)-mb*mb,0)
    cov=gaussian_valid(a*b)-ma*mb
    score=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return float(score.mean())


def metrics(original,reconstructed):
    return [{'name':name,'z_mm':float(z*1e3),'pearson':pearson(a,b),'ssim':ssim(a,b),
             'mse':float(np.mean((a-b)**2))} for name,z,a,b in zip(NAMES,ZS,original,reconstructed)]


def encode_phase(phase):
    wrapped=(phase+np.pi)%(2*np.pi)-np.pi
    return np.rint((wrapped+np.pi)/(2*np.pi)*255).astype(np.uint8)


def decode_phase(pixels):
    return pixels.astype(np.float64)/255*(2*np.pi)-np.pi


def phase_metrics(phase, original, c):
    """Evaluate the quantized mask with the same exposure as saved PNGs."""
    decoded=decode_phase(encode_phase(phase))
    raw=np.abs(propagate(decoded,c))**2*original.mean(axis=(1,2),keepdims=True)
    pixels=np.rint(np.clip(raw,0,1)*255)/255.
    return metrics(original,pixels)


def print_metrics(result, title='Final metrics from saved 8-bit phase mask'):
    rows=result['metrics_saved_png']
    print('\n'+title,flush=True)
    print(f'{"Image":<24} {"z (mm)":>9} {"Pearson":>11} {"SSIM":>11} {"MSE":>12}',flush=True)
    for m in rows:
        pcc=f'{m["pearson"]:.6f}' if m['pearson'] is not None else 'undefined'
        print(f'{m["name"]:<24} {m["z_mm"]:>9.3f} {pcc:>11} {m["ssim"]:>11.6f} {m["mse"]:>12.7f}',flush=True)
    valid=[m['pearson'] for m in rows if m['pearson'] is not None]
    mean_pcc=float(np.mean(valid)) if valid else float('nan')
    print(f'MEAN Pearson={mean_pcc:.6f}  SSIM={np.mean([m["ssim"] for m in rows]):.6f}  MSE={np.mean([m["mse"] for m in rows]):.7f}',flush=True)
    if 'loss_8bit_phase' in result:
        print(f'Amplitude loss={result["loss_8bit_phase"]:.6f}',flush=True)
    print('',flush=True)


def save_result(out,phase,original,c):
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'phase_continuous_rad.npy',(phase+np.pi)%(2*np.pi)-np.pi)
    Image.fromarray(encode_phase(phase)).save(out/'phase_mask.png')
    # Reopen actual saved 8-bit phase PNG, then reconstruct from decoded values.
    with Image.open(out/'phase_mask.png') as im:
        decoded=decode_phase(np.asarray(im))
    np.save(out/'phase_rad.npy',decoded)
    np.savetxt(out/'phase_rad.csv',decoded,delimiter=',',fmt='%.9f')
    np.savetxt(out/'phase_rad.txt',decoded,delimiter=' ',fmt='%.9f')
    raw_intensity=np.abs(propagate(decoded,c))**2
    # A fixed, declared exposure: target mean / model mean (=1 by Parseval).
    # No fitted gain, histogram adjustment or per-image min-max rescaling.
    exposure=original.mean(axis=(1,2))
    raw=raw_intensity*exposure[:,None,None]
    pixels=np.rint(np.clip(raw,0,1)*255).astype(np.uint8)
    np.save(out/'reconstructed_intensity_normalized.npy',raw_intensity)
    np.save(out/'reconstructed_brightness_unclipped.npy',raw)
    for name,a in zip(NAMES,pixels):
        Image.fromarray(a).save(out/f'{name}_reconstructed.png')
    result={'loss_continuous':objective(phase,original/exposure[:,None,None],c,False),
            'loss_8bit_phase':objective(decoded,original/exposure[:,None,None],c,False),
            'metrics_saved_png':metrics(original,pixels/255.),'metrics_unclipped_float':metrics(original,raw),
            'fraction_clipped_above_1':np.mean(raw>1,axis=(1,2)).tolist(),
            'output_pixel_pitch_um':(WL*ZS/(N*DX)*1e6).tolist(),
            'output_field_width_mm':(WL*ZS/DX*1e3).tolist(),
            'cross_target_pearson':[ [pearson(a,b) for b in original] for a in raw]}
    write_json(out/'metrics.json',result)
    return result


def checks():
    rng=np.random.default_rng(751)
    n=8; c=chirps(n); phi=rng.uniform(-np.pi,np.pi,(n,n)); target=rng.random((4,n,n))+.1
    target/=target.mean(axis=(1,2),keepdims=True)
    value,g=objective(phi,target,c)
    gradient_errors=[]
    for p in [(0,0),(1,4),(5,6),(7,7)]:
        plus=phi.copy(); minus=phi.copy(); plus[p]+=1e-5; minus[p]-=1e-5
        numerical=(objective(plus,target,c,False)-objective(minus,target,c,False))/2e-5
        gradient_errors.append(abs(g[p]-numerical)/max(1e-8,abs(g[p]),abs(numerical)))
    u=propagate(phi,c)
    energy_error=float(np.max(np.abs(np.mean(np.abs(u)**2,axis=(1,2))-1)))
    global_error=float(np.max(np.abs(np.abs(u)**2-np.abs(propagate(phi+.71,c))**2)))
    q=np.arange(n)-n//2
    dft=np.exp(-2j*np.pi*np.outer(q,q)/n)/np.sqrt(n)
    direct=dft@(np.exp(1j*phi)*c[0])@dft.T
    dft_error=float(np.max(np.abs(direct-u[0])))
    lens_error=float(abs(np.abs(propagate(-np.angle(c[0]),c)[0,n//2,n//2])**2-n*n))
    model=MLP(2); x=rng.normal(size=1024)
    test_g=rng.normal(size=(N,N))*.00001
    model.forward(x); grads=model.backward(test_g)
    mlp_errors=[]
    for weights,grad,idx in [(model.w1,grads[0],(2,47)),(model.w2,grads[1],(129,3))]:
        v=weights[idx]; weights[idx]=v+1e-5; a=float(np.sum(model.forward(x)*test_g))
        weights[idx]=v-1e-5; b=float(np.sum(model.forward(x)*test_g)); weights[idx]=v
        num=(a-b)/2e-5
        mlp_errors.append(abs(grad[idx]-num)/max(1e-9,abs(num),abs(grad[idx])))
    image=rng.random((128,128))
    phase_roundtrip=float(np.max(np.abs(np.angle(np.exp(1j*(decode_phase(encode_phase(phi))-phi))))))
    out={'fresnel_gradient_max_relative_error':max(gradient_errors),
         'mlp_gradient_max_relative_error':max(mlp_errors),'parseval_error':energy_error,
         'global_phase_invariance_error':global_error,'direct_dft_field_error':dft_error,
         'quadratic_lens_peak_error':lens_error,'ssim_self':ssim(image,image),
         'pearson_self':pearson(image,image),'phase_quantization_max_error_rad':phase_roundtrip}
    assert max(gradient_errors)<1e-5 and max(mlp_errors)<1e-5
    assert max(energy_error,global_error,dft_error,lens_error)<1e-10
    assert abs(out['ssim_self']-1)<1e-10 and abs(out['pearson_self']-1)<1e-10
    assert phase_roundtrip<=np.pi/255+1e-12
    out['passed']=True
    return out


def fit(original,steps,seeds,out):
    x=features(original); target=original/original.mean(axis=(1,2),keepdims=True); c=chirps()
    history=[]; summary=[]; best_mlp=None; best_direct=None; initial_best=None
    started=time.perf_counter()
    for seed in range(seeds):
        model=MLP(seed)
        initial=model.forward(x).copy()
        for kind in ['mlp','direct_phase']:
            if kind=='mlp':
                optim=Adam([model.w1,model.w2],[.0002,.025])
            else:
                phase=initial.copy(); optim=Adam([phase],[.06])
            best_loss=float('inf'); best_state=None
            for step in range(steps+1):
                if kind=='mlp': phase=model.forward(x)
                loss,g=objective(phase,target,c)
                if loss<best_loss:
                    best_loss=loss
                    best_state=(model.w1.copy(),model.w2.copy()) if kind=='mlp' else phase.copy()
                if step%100==0 or step==steps:
                    history.append({'kind':kind,'seed':seed,'step':step,'loss':loss,'elapsed_s':time.perf_counter()-started})
                    current=phase_metrics(phase,original,c)
                    print(f'{kind} seed={seed} step={step}/{steps} loss={loss:.6f} '
                          f'Pearson={np.mean([m["pearson"] for m in current]):.5f} '
                          f'SSIM={np.mean([m["ssim"] for m in current]):.5f}',flush=True)
                if step==steps: break
                factor=.15+.85*.5*(1+math.cos(math.pi*step/steps))
                optim.step(model.backward(g) if kind=='mlp' else [g],factor)
            summary.append({'kind':kind,'seed':seed,'initial_loss':objective(initial,target,c,False),'best_loss':best_loss})
            if kind=='mlp' and (best_mlp is None or best_loss<best_mlp[0]):
                best_mlp=(best_loss,best_state,seed); initial_best=initial.copy()
            if kind=='direct_phase' and (best_direct is None or best_loss<best_direct[0]):
                best_direct=(best_loss,best_state,seed)
    model.w1,model.w2=best_mlp[1]
    model.save(out/'model.npz')
    trained=model.forward(x).copy()
    results=save_result(out,trained,original,c)
    random_result=save_result(out/'baseline_untrained',initial_best,original,c)
    direct_result=save_result(out/'baseline_direct_phase',best_direct[1],original,c)
    # Sensitivity probes only. These are not held-out-set generalization tests.
    swapped=original[[1,0,2,3]]
    swapped_phase=model.forward(features(swapped)).copy()
    blank_phase=model.forward(features(np.full_like(original,.5))).copy()
    sensitivity={
        'swapped_first_two_inputs_loss_against_swapped_targets':objective(swapped_phase,swapped/swapped.mean(axis=(1,2),keepdims=True),c,False),
        'swapped_first_two_inputs_phase_circular_rms_change':float(np.sqrt(np.mean(np.angle(np.exp(1j*(swapped_phase-trained)))**2))),
        'blank_input_phase_circular_rms_change':float(np.sqrt(np.mean(np.angle(np.exp(1j*(blank_phase-trained)))**2))),
        'original_inputs_loss':results['loss_continuous'],
        'interpretation':'Sensitivity is not evidence of successful prediction for new inputs; only one example was fitted.'}
    model.forward(x)
    reloaded=MLP.load(out/'model.npz').forward(x)
    assert np.max(np.abs(reloaded-trained))<1e-10
    report={'experiment':'one fixed 4-image example, training-set fit; no held-out test',
        'architecture':'MLP: 1024 -> tanh(32) -> 16384 raw phases, no biases',
        'parameter_count':int(model.w1.size+model.w2.size),'steps_per_run':steps,'seeds':seeds,
        'best_mlp_seed':best_mlp[2],'best_direct_seed':best_direct[2],
        'selection':'minimum continuous amplitude training loss; no validation set',
        'results':results,'untrained':random_result,'direct_phase':direct_result,
        'runs':summary,'sensitivity_probes':sensitivity,'elapsed_s':time.perf_counter()-started,
        'python':sys.version,'numpy':np.__version__}
    write_json(out/'experiment.json',report)
    with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    make_visuals(original,out,history)
    return report


def make_visuals(original,out,history):
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),18) if fontpath.exists() else ImageFont.load_default()
    canvas=Image.new('RGB',(1220,4*300+90),'#f3f4f6'); draw=ImageDraw.Draw(canvas)
    draw.text((16,14),'Single-example fit | reconstructions from saved 8-bit phase masks',font=font,fill='black')
    for j,label in enumerate(['Target','Untrained MLP','Trained MLP','Direct-phase baseline']):
        draw.text((20+j*305,51),label,font=font,fill='black')
    for i,name in enumerate(NAMES):
        y=90+i*300
        ims=[Image.fromarray(np.rint(original[i]*255).astype(np.uint8))]
        for folder in [out/'baseline_untrained',out,out/'baseline_direct_phase']:
            with Image.open(folder/f'{name}_reconstructed.png') as im: ims.append(im.copy())
        for j,im in enumerate(ims):
            canvas.paste(im.resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(20+j*305,y))
        draw.text((20,y+262),f'{name} | z={ZS[i]*1000:.1f} mm',font=font,fill='black')
    canvas.save(out/'comparison.png')
    chart=Image.new('RGB',(1000,540),'white'); draw=ImageDraw.Draw(chart)
    left,top,right,bottom=80,55,965,470
    draw.text((20,12),'Amplitude loss vs optimizer steps (all restarts)',font=font,fill='black')
    ymax=max(h['loss'] for h in history)*1.05; xmax=max(h['step'] for h in history)
    for j in range(6):
        val=ymax*j/5; y=bottom-(bottom-top)*j/5
        draw.line((left,y,right,y),fill='#dddddd'); draw.text((8,y-10),f'{val:.2f}',font=font,fill='black')
    for kind,color in [('mlp','#2563eb'),('direct_phase','#e07020')]:
        for seed in sorted({h['seed'] for h in history}):
            rows=[h for h in history if h['kind']==kind and h['seed']==seed]
            points=[(left+(right-left)*h['step']/max(xmax,1),bottom-(bottom-top)*h['loss']/ymax) for h in rows]
            if len(points)>1: draw.line(points,fill=color,width=2)
    draw.text((left,490),f'0                      Blue: MLP | Orange: direct phase                     {xmax}',font=font,fill='black')
    chart.save(out/'training_curve.png')


def main():
    global NAMES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','check','infer','improve'],nargs='?',default='infer')
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'results'/'model.npz')
    parser.add_argument('--inputs',type=Path,nargs=4)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='check':
        print(json.dumps(checks(),indent=2)); return
    paths=args.inputs or [ROOT/'inputs'/f'{name}.png' for name in NAMES]
    if args.command=='improve':
        if args.steps<1: parser.error('steps must be positive')
        from improve import train_model
        train_model(load_images(paths),args.checkpoint,args.out or ROOT/'results_improved',args.steps)
    elif args.command=='train':
        if args.steps<1 or args.seeds<1: parser.error('steps and seeds must be positive')
        for name in NAMES:
            dest=ROOT/'inputs'/f'{name}.png'
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(ROOT.parent/'photonics_dataset'/'targets'/f'{name}.png',dest)
        out=args.out or ROOT/'results'; out.mkdir(parents=True,exist_ok=True)
        write_json(out/'checks.json',checks())
        images=load_images(paths)
        report=fit(images,args.steps,args.seeds,out)
        write_json(out/'input_hashes.json',[{'index':i,'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for i,p in enumerate(paths)])
        print_metrics(report['results'])
        print(f'Elapsed: {report["elapsed_s"]:.1f} s | Results: {out.resolve()}',flush=True)
    else:
        if args.inputs:
            labels=[p.stem for p in paths]
            NAMES=labels if len(set(labels))==4 else [f'{i+1:02d}_{p.stem}' for i,p in enumerate(paths)]
        images=load_images(paths); model=MLP.load(args.checkpoint)
        result=save_result(args.out or ROOT/'inference',model.forward(features(images)),images,chirps())
        print_metrics(result)
        print(f'Results: {(args.out or ROOT/"inference").resolve()}',flush=True)


if __name__=='__main__':
    main()
