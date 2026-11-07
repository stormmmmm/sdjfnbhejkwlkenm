"""Optimize a shared four-plane phase with structure and residual leakage losses.

No target images are blended into reconstructions; all evaluations propagate
the saved quantized phase. The fixed four-image fit does not generalize.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, json, time
from pathlib import Path
import numpy as np
from . import experiment as e


class Loss:
    def __init__(self,original,weights,correlation_floor=None,floor_weight=0.):
        self.original=original
        if np.any(np.std(original,axis=(1,2))<1e-8):
            raise ValueError('Для дообучения нужны изображения с ненулевым контрастом: Pearson для константы не определён.')
        self.mean=original.mean(axis=(1,2),keepdims=True)
        self.target=original/self.mean
        self.weights=weights
        self.correlation_floor=None if correlation_floor is None else np.asarray(correlation_floor).reshape(4,1,1)
        self.floor_weight=float(floor_weight)
        self.c=e.chirps()
        q=np.minimum(np.arange(128),128-np.arange(128))
        g=np.exp(-q*q/(2*1.5**2)); g[q>5]=0; g/=g.sum()
        self.kernel=np.fft.fft2(g[:,None]*g[None,:]).real
        self.tm=self.smooth(original)
        self.tv=self.smooth(original**2)-self.tm**2
        self.centered=original-original.mean(axis=(1,2),keepdims=True)
        self.targetnorm=np.sqrt(np.sum(self.centered**2,axis=(1,2),keepdims=True))
        self.basis=[]
        for i in range(4):
            own=self.centered[i]
            others=[]
            for j in range(4):
                if i==j: continue
                other=self.centered[j].copy()
                other-=np.sum(other*own)/np.sum(own**2)*own
                other/=max(np.sqrt(np.mean(other**2)),1e-12)
                others.append(other)
            self.basis.append(others)
        self.basis=np.asarray(self.basis)
        self.std=np.std(original,axis=(1,2))

    def smooth(self,a):
        return np.fft.ifft2(np.fft.fft2(a)*self.kernel).real

    def ssim_loss(self,p):
        m=self.smooth(p)
        v=self.smooth(p*p)-m*m
        cov=self.smooth(p*self.original)-m*self.tm
        a=2*m*self.tm+.01**2; b=2*cov+.03**2
        c=m*m+self.tm*self.tm+.01**2; d=v+self.tv+.03**2
        score=a*b/(c*d)
        da=b/(c*d); db=a/(c*d); dc=-a*b/(c*c*d); dd=-a*b/(c*d*d)
        gm=2*self.tm*da+2*m*dc
        gv=dd; gc=2*db
        grad=self.smooth(gm-2*m*gv-self.tm*gc)+2*p*self.smooth(gv)+self.original*self.smooth(gc)
        return float(1-score.mean()), -grad/p.size

    def leakage(self,p):
        # Nuisance targets have been made orthogonal to own target and DC.
        # Coefficients measure pattern leakage in units of own target std.
        coefficients=np.mean((p-self.original)[:,None]*self.basis,axis=(2,3))/self.std[:,None]
        return coefficients

    def evaluate(self,phase,gradient=True):
        u=e.propagate(phase,self.c)
        intensity=np.abs(u)**2
        p=intensity*self.mean
        total=0.; gp=np.zeros_like(p)
        mse,pear,ssim,ghost=self.weights
        if mse:
            scale=np.mean(self.original**2,axis=(1,2),keepdims=True)
            diff=p-self.original
            total+=mse*np.mean(diff*diff/scale)
            gp+=mse*2*diff/(scale*p.size)
        if pear:
            centered=p-p.mean(axis=(1,2),keepdims=True)
            norm=np.sqrt(np.sum(centered**2,axis=(1,2),keepdims=True)+1e-16)
            correlation=np.sum(centered*self.centered,axis=(1,2),keepdims=True)/(norm*self.targetnorm)
            total+=pear*(1-float(correlation.mean()))
            gp-=pear/4*(self.centered/(norm*self.targetnorm)-correlation*centered/(norm*norm))
            if self.correlation_floor is not None and self.floor_weight:
                deficit=np.maximum(self.correlation_floor-correlation,0)
                total+=self.floor_weight*float(np.mean(deficit**2))
                gp-=self.floor_weight/2*deficit*(self.centered/(norm*self.targetnorm)-correlation*centered/(norm*norm))
        if ssim:
            val,grad=self.ssim_loss(p); total+=ssim*val; gp+=ssim*grad
        if ghost:
            coef=self.leakage(p)
            # Square penalizes positive ghosts and negative imprints alike.
            total+=ghost*np.mean(coef**2)
            gp+=ghost*2*np.sum(coef[:,:,None,None]*self.basis,axis=1)/(12*128*128*self.std[:,None,None])
        if not gradient: return float(total)
        di=gp*self.mean
        back=np.sum(np.conj(self.c)*e.ifft(di*u),axis=0)
        return float(total),2*np.imag(np.exp(-1j*phase)*back)


def summary(phase,original,loss):
    quant=e.decode_phase(e.encode_phase(phase))
    raw=np.abs(e.propagate(quant,loss.c))**2*loss.mean
    pixels=np.rint(np.clip(raw,0,1)*255)/255.
    values=e.metrics(original,pixels)
    leaks=loss.leakage(raw)
    rawmetrics=e.metrics(original,raw)
    pc=raw-raw.mean(axis=(1,2),keepdims=True)
    tc=original-original.mean(axis=(1,2),keepdims=True)
    own_gain=np.sum(pc*tc,axis=(1,2))/np.sum(tc**2,axis=(1,2))
    return {'mean_pearson':float(np.mean([x['pearson'] for x in values])),
            'mean_ssim':float(np.mean([x['ssim'] for x in values])),
            'mean_mse':float(np.mean([x['mse'] for x in values])),
            'mean_raw_pearson':float(np.mean([x['pearson'] for x in rawmetrics])),
            'mean_raw_ssim':float(np.mean([x['ssim'] for x in rawmetrics])),
            'leakage_rms':float(np.sqrt(np.mean(leaks**2))),
            'positive_leakage_mean':float(np.maximum(leaks,0).mean()),
            'leakage_coefficients':leaks.tolist(),'own_contrast_gain':own_gain.tolist(),
            'metrics':values}


def check_gradients(original):
    rng=np.random.default_rng(59)
    phase=rng.uniform(-np.pi,np.pi,(128,128))
    errors=[]
    for weights in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
        loss=Loss(original,weights); val,g=loss.evaluate(phase)
        # Directional check is more robust than a pixel with near-zero derivative.
        direction=rng.normal(size=(128,128)); direction/=np.linalg.norm(direction)
        eps=1e-4
        numerical=(loss.evaluate(phase+eps*direction,False)-loss.evaluate(phase-eps*direction,False))/(2*eps)
        exact=float(np.sum(g*direction))
        err=abs(exact-numerical)/max(1e-9,abs(exact),abs(numerical))
        errors.append(err)
    assert max(errors)<1e-4,errors
    return errors


def run(original,start,weights,steps,lr,out):
    out.mkdir(parents=True,exist_ok=True)
    loss=Loss(original,weights)
    phase=start.copy(); optimizer=e.Adam([phase],[lr])
    history=[]; best=float('inf'); bestphase=None
    for step in range(steps+1):
        value,g=loss.evaluate(phase)
        if value<best: best=value; bestphase=phase.copy()
        if step%200==0 or step==steps:
            stats=summary(phase,original,loss)
            row={'step':step,'loss':value,**stats}; history.append(row)
            print(f'{out.name} {step}/{steps}: loss={value:.5f} PCC={stats["mean_pearson"]:.4f} SSIM={stats["mean_ssim"]:.4f} leak={stats["leakage_rms"]:.4f}',flush=True)
        if step==steps: break
        factor=.1+.9*.5*(1+np.cos(np.pi*step/steps))
        optimizer.step([g],factor)
    np.save(out/'candidate_phase.npy',bestphase)
    stats=summary(bestphase,original,loss)
    e.write_json(out/'optimization.json',{'weights':weights,'steps':steps,'learning_rate':lr,'summary':stats,'history':history})
    e.save_result(out,bestphase,original,loss.c)
    return bestphase,stats


def train_model(original,checkpoint,out,steps=2000):
    """Fine-tune MLP through physics; retain per-plane correlation floors.

    This remains one-example fitting. No target blending or output filtering.
    """
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    model=e.MLP.load(checkpoint)
    x=e.features(original)
    start=model.forward(x).copy()
    raw=np.abs(e.propagate(start,e.chirps()))**2*original.mean(axis=(1,2),keepdims=True)
    floors=[e.pearson(a,b)+.015 for a,b in zip(original,raw)]
    loss=Loss(original,(.3,1,.5,.6),floors,40.)
    before=summary(start,original,loss)
    opt=e.Adam([model.w1,model.w2],[.00005,.015])
    best=float('inf'); chosen=None; history=[]
    started=time.perf_counter()
    for step in range(steps+1):
        phase=model.forward(x)
        value,g=loss.evaluate(phase)
        if value<best:
            best=value; chosen=(model.w1.copy(),model.w2.copy())
        if step%100==0 or step==steps:
            stats=summary(phase,original,loss)
            history.append({'step':step,'loss':value,**stats})
            print(f'MLP improve {step}/{steps} loss={value:.6f} '
                  f'Pearson={stats["mean_pearson"]:.5f} SSIM={stats["mean_ssim"]:.5f} '
                  f'leakage_RMS={stats["leakage_rms"]:.5f}',flush=True)
        if step==steps: break
        factor=.1+.9*.5*(1+np.cos(np.pi*step/steps))
        opt.step(model.backward(g),factor)
    model.w1,model.w2=chosen
    model.save(out/'model.npz')
    phase=model.forward(x)
    result=e.save_result(out,phase,original,loss.c)
    after=summary(phase,original,loss)
    e.write_json(out/'improvement.json',{'method':'physics-supervised MLP fine tuning',
        'training_examples':1,'held_out_examples':0,'steps':steps,
        'selection':'minimum combined loss, fixed training example',
        'weights':loss.weights,'correlation_floor':floors,'floor_weight':40,
        'before':before,'after':after,'elapsed_s':time.perf_counter()-started,
        'history':history,'checkpoint_sha256':__import__('hashlib').sha256(Path(checkpoint).read_bytes()).hexdigest()})
    e.print_metrics(result,'Improved MLP: metrics after 8-bit phase quantization')
    print(f'Leakage RMS: {before["leakage_rms"]:.5f} -> {after["leakage_rms"]:.5f}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=1400)
    p.add_argument('--out',type=Path,default=e.ROOT/'improvement_trials')
    p.add_argument('--check',action='store_true')
    p.add_argument('--start',type=Path,default=e.ROOT/'results'/'phase_continuous_rad.npy')
    p.add_argument('--weights',type=float,nargs=4)
    args=p.parse_args()
    original=e.load_images([e.ROOT/'inputs'/f'{name}.png' for name in e.NAMES])
    if args.check:
        print(check_gradients(original)); return
    start=np.load(args.start)
    configs={'intensity':(1,0,0,0),'pearson_structure':(.3,1,.5,0),
             'suppress_ghosts':(.3,1,.5,2),'strong_structure':(.2,.5,2,1)}
    if args.weights: configs={'custom':tuple(args.weights)}
    args.out.mkdir(parents=True,exist_ok=True)
    e.write_json(args.out/'before.json',summary(start,original,Loss(original,(1,1,1,1))))
    for label,weights in configs.items():
        run(original,start,weights,args.steps,.045,args.out/label)


if __name__=='__main__': main()
