from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
import sim_langevin as L
import numpy as np, glob, os, pandas as pd
L.CFG.output_root=_SAL+"/sim_CGM_diab"
L.CFG.n_trials=120; L.CFG.limit_days=12
os.makedirs(L.CFG.output_root,exist_ok=True)
files=sorted(glob.glob("cgm_diab_in/subject_*_E.csv"),key=lambda f:int(os.path.basename(f).split('_')[1]))
rows=[]
for fp in files:
    r=L.process_subject(fp,verbose=False)
    if not r: continue
    d=r['diagnostics'];p=r['params']
    rows.append(dict(ID=r['ID'],
        corr=round(d['metrics']['corr'],3),rmse=round(d['metrics']['rmse'],1),
        H_real=round(d['hxc']['H_real'],3),H_sim=round(d['hxc']['H_sim'],3),
        C_real=round(d['hxc']['C_real'],3),C_sim=round(d['hxc']['C_sim'],3),
        beta_real=round(d['spectral']['beta_real'],2),beta_sim=round(d['spectral']['beta_sim'],2),
        TIR_real=round(d['clinical']['real']['tir'],3),TIR_sim=round(d['clinical']['sim']['tir'],3),
        alpha=round(p['alpha'],5),tau=round(p['tau'],1),k_f=round(p['k_f'],2),beta=round(p['beta'],3)))
    print(f"CGM-{r['ID']}: corr={rows[-1]['corr']:.2f} H {rows[-1]['H_real']:.2f}->{rows[-1]['H_sim']:.2f} TIR {rows[-1]['TIR_real']:.2f}->{rows[-1]['TIR_sim']:.2f}")
R=pd.DataFrame(rows);R.to_csv("sim_CGM_diab_resumen.csv",index=False)
print("\ncorr medio %.3f | H %.3f->%.3f | C %.3f->%.3f | TIR %.3f->%.3f | beta %.2f->%.2f"%(
 R['corr'].mean(),R.H_real.mean(),R.H_sim.mean(),R.C_real.mean(),R.C_sim.mean(),R.TIR_real.mean(),R.TIR_sim.mean(),R.beta_real.mean(),R.beta_sim.mean()))
