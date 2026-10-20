from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
import os, glob, math, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE=_CGM; M=3; TAU=1

# ---------- funciones EXACTAS del notebook ----------
def bandt_pompe_pdf(x,m=3,tau=1):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    n=len(x)-(m-1)*tau
    patterns=list(itertools.permutations(range(m))); idx={p:i for i,p in enumerate(patterns)}
    counts=np.zeros(math.factorial(m))
    for t in range(n):
        v=x[t:t+m*tau:tau]; counts[idx[tuple(np.argsort(v,kind="mergesort"))]]+=1
    return counts/counts.sum()
def _S(P):
    P=np.asarray(P,float); nz=P[P>0]; return float(-np.sum(nz*np.log(nz))) if len(nz) else 0.0
def q0_norm(N): return -2.0/(((N+1)/N)*np.log(N+1)-2*np.log(2*N)+np.log(N))
def HxC_from_signal(x,m=3,tau=1):
    P=bandt_pompe_pdf(x,m,tau); N=len(P); Pe=np.ones(N)/N
    H=_S(P)/np.log(N); JSD=_S(0.5*(P+Pe))-0.5*_S(P)-0.5*_S(Pe)
    return H, H*JSD*q0_norm(N)
def get_complexity_bounds(m=3):
    N=math.factorial(m)
    def sc(P):
        Pe=np.ones(N)/N; nz=P[P>0]
        if len(nz)==0: return 0.,0.
        H=-np.sum(nz*np.log(nz))/np.log(N); Pm=0.5*(P+Pe)
        def S(p): pn=p[p>0]; return -np.sum(pn*np.log(pn))
        JSD=S(Pm)-0.5*S(P)-0.5*S(Pe); Q0=-2.0/((N+1)/N*np.log(N+1)-2*np.log(2*N)+np.log(N))
        return H,H*JSD*Q0
    Hmin,Cmin,Hmax,Cmax=[],[],[],[]
    for p in np.linspace(0.001,0.999,200):
        P=np.array([p]+list((1-p)/(N-1)*np.ones(N-1))); h,c=sc(P); Hmax.append(h); Cmax.append(c)
    for k in range(1,N):
        for p in np.linspace(1/k,1/(k+1),50):
            P=np.zeros(N); P[:k]=p
            if 1-k*p>0: P[k]=1-k*p
            h,c=sc(P); Hmin.append(h); Cmin.append(c)
    return Hmin,Cmin,Hmax,Cmax

# ---------- clasificación clínica (A1c / ayunas) ----------
b=pd.read_csv(f"{BASE}/bio.csv")
b['A1c']=pd.to_numeric(b['A1c PDL (Lab)'],errors='coerce'); b['F']=pd.to_numeric(b['Fasting GLU - PDL (Lab)'],errors='coerce')
def clin(r):
    a,f=r['A1c'],r['F']
    if (pd.notna(a) and a>=6.5) or (pd.notna(f) and f>=126): return 'Diabético'
    if (pd.notna(a) and a>=5.7) or (pd.notna(f) and f>=100): return 'Prediabético'
    return 'Sano'
cmap={int(r['subject']):clin(r) for _,r in b.iterrows()}

# ---------- H×C sobre Libre COMPLETO (15 min nativo) ----------
rows=[]
for d in sorted(glob.glob(f"{BASE}/CGMacros-*")):
    subj=os.path.basename(d); sid=int(subj.split('-')[1])
    df=pd.read_csv(f"{d}/{subj}.csv",usecols=['Timestamp','Libre GL']); df['Timestamp']=pd.to_datetime(df['Timestamp'])
    s=df.set_index('Timestamp')['Libre GL'].resample('15min').mean().interpolate('time',limit_direction='both')
    H,C=HxC_from_signal(s.values,M,TAU)
    rows.append(dict(subject=subj,grupo=cmap[sid],N=len(s),H=round(H,4),C=round(C,4)))
R=pd.DataFrame(rows); R.to_csv(_SAL+"/HxC_libre_full.csv",index=False)

# ---------- figura ----------
Hmin,Cmin,Hmax,Cmax=get_complexity_bounds(M)
COL={'Sano':'#2e9e4f','Prediabético':'#e8a020','Diabético':'#e0342a'}
MK ={'Sano':'o','Prediabético':'s','Diabético':'D'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,ax=plt.subplots(figsize=(10,7.5))
ax.plot(Hmin,Cmin,color='k',alpha=0.55,lw=1.4); ax.plot(Hmax,Cmax,color='k',alpha=0.55,lw=1.4)
for g in ['Sano','Prediabético','Diabético']:
    x=R[R.grupo==g]
    ax.scatter(x.H,x.C,s=95,c=COL[g],marker=MK[g],edgecolor='k',lw=.5,alpha=.9,label=f'{g} (n={len(x)})',zorder=5)
    # centroide
    ax.scatter(x.H.mean(),x.C.mean(),s=340,c=COL[g],marker='X',edgecolor='white',lw=2,zorder=8)
ax.set_xlabel('H  (entropía de permutación normalizada)',fontsize=12)
ax.set_ylabel('C  (complejidad estadística)',fontsize=12)
ax.set_title('Plano H×C sobre Libre completo (M=3, τ=1)\ncon fronteras C$_{min}$/C$_{max}$',fontsize=13,weight='bold')
ax.set_xlim(0.4,1.0); ax.set_ylim(0,0.30)
ax.legend(loc='upper right',fontsize=11,framealpha=.95); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(_SAL+"/plano_HC_libre_full.png",dpi=150)

print(R.groupby('grupo')[['H','C']].agg(['mean','std','count']).round(3).to_string())
print("\nfigura ok")
