"""Would a third axis help if the LABELS were spatially coherent?

The frontend test is confounded: their `vent` spans z -1.50..1.29, so elevation
cannot separate it no matter what lz is used. This substitutes an ORACLE
frontend -- the GT eval points themselves, subsampled -- so every label is
spatially where it belongs. If 3D beats 2D here, elevation is worth having once
the frontend improves. If not, it is genuinely not needed for this task.
"""
import importlib.util, sys, types, numpy as np, statistics as st
sys.path.insert(0, "."); 
try: import torch
except ModuleNotFoundError:
    class _A(types.ModuleType):
        def __getattr__(s,n): raise RuntimeError("torch")
    sys.modules["torch"]=_A("torch")
from vsa_cognitive_mapping.vsa import Phasor
spec=importlib.util.spec_from_file_location("s","student_gpu_package/05_score.py")
m=importlib.util.module_from_spec(spec); sys.argv=["x"]; spec.loader.exec_module(m)
HD=4096; CAP=400

def run(P, cls, Q, names, bases, scales, rng):
    sem={c:Phasor(dim=HD,seed=9000+i).values for i,c in enumerate(names)}
    tr=np.zeros(HD,np.complex128)
    for c in names:
        idx=np.flatnonzero(cls==c)
        if not len(idx): continue
        if len(idx)>CAP: idx=rng.choice(idx,CAP,replace=False)
        acc=np.ones((len(idx),HD),np.complex128)
        for a,(B,l) in enumerate(zip(bases,scales)): acc*=B[None,:]**(P[idx,a,None]/l)
        tr+=sem[c]*(acc.sum(0)/len(idx))
    tr/=max(np.abs(tr).max(),1e-12)
    V=np.stack([tr/sem[c] for c in names]); F=np.empty((len(names),len(Q)))
    for i0 in range(0,len(Q),2048):
        i1=min(i0+2048,len(Q)); g=np.ones((i1-i0,HD),np.complex128)
        for a,(B,l) in enumerate(zip(bases,scales)): g*=B[None,:]**(Q[i0:i1,a,None]/l)
        F[:,i0:i1]=(V@np.conj(g).T).real
    return np.array([names[w] for w in F.argmax(0)])

for s in ["room0","room2"]:
    E=np.load(f"student_gpu_package/handoff/{s}_cgfront/eval_points.npz",allow_pickle=True)
    xyz,gt=E["xyz"],E["gt_class"].astype(str)
    v=xyz.var(0); a,b=sorted(np.argsort(v)[-2:]); up=({0,1,2}-{a,b}).pop()
    rng=np.random.default_rng(0)
    # ORACLE frontend: half the eval points, with their true labels
    sel=rng.random(len(xyz))<0.5
    P,cls=xyz[sel],gt[sel]; Q=xyz
    names=sorted(set(cls))
    hi=sorted({c for c in set(gt)-set(m.CG_EXCLUDE_6)
               if (gt==c).sum()>=5 and xyz[gt==c,up].mean()>xyz[:,up].mean()+0.5})
    Bx=Phasor(dim=HD,seed=11).values; By=Phasor(dim=HD,seed=12).values
    Bz=Phasor(dim=HD,seed=13).values
    print(f"\n{s}: oracle frontend {sel.sum()} pts, {len(names)} classes; high={hi}")
    print(f"  {'lz':>6}{'mAcc':>8}{'mPrec':>8}{'mF1':>8}{'high':>8}")
    for lz in [None,0.15,0.3,0.6,1.2,2.5]:
        if lz is None:
            pr=run(P[:,[a,b]],cls,Q[:,[a,b]],names,[Bx,By],[0.45,0.27],rng); tag="2D"
        else:
            pr=run(P[:,[a,b,up]],cls,Q[:,[a,b,up]],names,[Bx,By,Bz],[0.45,0.27,lz],rng)
            tag=f"{lz:.2f}"
        d=m.macc_full(gt,pr,exclude=m.CG_EXCLUDE_6)
        h=st.mean(float((pr[gt==c]==c).mean()) for c in hi) if hi else float("nan")
        print(f"  {tag:>6}{d['macc']:>8.3f}{d['mprec']:>8.3f}{d['mf1']:>8.3f}{h:>8.3f}",flush=True)
