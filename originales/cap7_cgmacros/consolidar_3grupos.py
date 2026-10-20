from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
"""Detección uniforme (ingestas niveles + alba ΔG>=30, corte 8:00) en los 45 sujetos, agregada por grupo."""
import pandas as pd, numpy as np, glob, os, pywt, math, itertools
STDIR=_SAL+"/S_t"; BASE=_CGM
DT_H=15/60; EPS=1e-9
L_H=2.5; DGMIN=20.0; QMIN=0.2; DAWN_DGMIN=30.0
CLASS=[('fuerte',120,0.70,0.65,0.15),('moderada',70,0.55,0.35,0.0),('leve',25,0.0,0.0,0.0)]; TOL=0.75
def ipn(y):
    Ap=DT_H*np.sum(np.maximum(y,0));An=DT_H*np.sum(np.maximum(-y,0));return (Ap-An)/(Ap+An+EPS)
def wen(S,band):
    per=np.linspace(band[0],band[1],12);fc=pywt.central_frequency('cmor1.5-1.0')
    sc=fc/((1/per)*DT_H);coef,_=pywt.cwt(S,sc,'cmor1.5-1.0',sampling_period=DT_H);return np.mean(np.abs(coef)**2,axis=0)
def classify(dg,iG,iS,wmax):
    for n,dgt,igt,ist,wt in CLASS:
        if dg>=dgt and iG>igt and iS>ist and wmax>wt: return n
    return None
def dawn(t,G,S,Ws):
    df=pd.DataFrame({'t':t,'G':G,'S':S,'W':Ws/(Ws.max()+EPS)});df['d']=df.t.dt.date;df['m']=df.t.dt.hour*60+df.t.dt.minute;ev=[]
    for d,g in df.groupby('d'):
        w=g[(g.m>=180)&(g.m<=480)]
        if len(w)<8:continue
        G0=w.G.iloc[0];dg=w.G.iloc[-1]-G0;iG=ipn((w.G-G0).values);iS=ipn(w.S.values)
        if dg>DAWN_DGMIN and iG>0 and iS>0: ev.append(dict(t_ini=w.t.iloc[0],t_fin=w.t.iloc[-1]))
    return ev
def meals(t,G,S,Wf,dw):
    N=len(G);Lw=int(round(L_H/DT_H));Wn=Wf/(Wf.max()+EPS);cd=[]
    for a in range(0,N-Lw):
        b=a+Lw;Gw=G[a:b];Sw=S[a:b];dg=Gw.max()-Gw[0]
        if dg<DGMIN:continue
        iG=ipn(Gw-Gw[0]);iS=ipn(Sw);wmax=Wn[a:b].max()
        Q=(dg/40)*(max(iS,0)+EPS)*(1+min(wmax,1.5))*(max(iG,0)+EPS)
        if Q<QMIN:continue
        cd.append(dict(a=a,b=b,dg=dg,iG=iG,iS=iS,wmax=wmax,Q=Q))
    cd.sort(key=lambda c:-c['Q']);kept=[]
    for c in cd:
        if all(c['b']<=k['a'] or k['b']<=c['a'] for k in kept):
            ti=t.iloc[c['a']];tf=t.iloc[min(c['b'],N-1)]
            if any(not(tf<=e['t_ini'] or e['t_fin']<=ti) for e in dw) and 3<=ti.hour<8: continue
            lab=classify(c['dg'],c['iG'],c['iS'],c['wmax'])
            if lab: c['clase']=lab;c['t_ini']=ti;c['t_fin']=tf;kept.append(c)
    return kept
# H×C exacto notebook
def hxc(x,m=3,tau=1):
    x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x)-(m-1)*tau
    pats=list(itertools.permutations(range(m)));idx={p:i for i,p in enumerate(pats)};c=np.zeros(math.factorial(m))
    for tt in range(n): c[idx[tuple(np.argsort(x[tt:tt+m*tau:tau],kind='mergesort'))]]+=1
    P=c/c.sum();N=len(P);Pe=np.ones(N)/N
    def S(p):p=p[p>0];return -np.sum(p*np.log(p))
    H=S(P)/np.log(N);JSD=S(0.5*(P+Pe))-0.5*S(P)-0.5*S(Pe);Q0=-2/(((N+1)/N)*np.log(N+1)-2*np.log(2*N)+np.log(N))
    return H,H*JSD*Q0
b=pd.read_csv(f"{BASE}/bio.csv");b['A']=pd.to_numeric(b['A1c PDL (Lab)'],errors='coerce');b['F']=pd.to_numeric(b['Fasting GLU - PDL (Lab)'],errors='coerce')
def cl(r):
    if (pd.notna(r['A']) and r['A']>=6.5) or (pd.notna(r['F']) and r['F']>=126):return 'Diabético'
    if (pd.notna(r['A']) and r['A']>=5.7) or (pd.notna(r['F']) and r['F']>=100):return 'Prediabético'
    return 'Sano'
b['g']=b.apply(cl,axis=1);gmap={int(r['subject']):r['g'] for _,r in b.iterrows()}
rows=[]
for d in sorted(glob.glob(f"{BASE}/CGMacros-*")):
    subj=os.path.basename(d);sid=int(subj.split('-')[1])
    st=pd.read_csv(f"{STDIR}/{subj}_St.csv",parse_dates=['Timestamp']);t=st.Timestamp;G=st.G.values;S=st.S.values
    Wf=wen(S,(0.5,2.0));Ws=wen(S,(3.0,8.0));dw=dawn(t,G,S,Ws);mk=meals(t,G,S,Wf,dw)
    ndays=t.dt.date.nunique()
    ann=pd.read_csv(f"{d}/{subj}.csv",usecols=['Timestamp','Meal Type']);ann['Timestamp']=pd.to_datetime(ann['Timestamp']);ann=ann[ann['Meal Type'].notna()];mt=ann['Timestamp'].values
    hit=np.zeros(len(ann),bool);dethit=0
    lvl={'fuerte':0,'moderada':0,'leve':0}
    for k in mk:
        lvl[k['clase']]+=1
        lo=np.datetime64(k['t_ini'])-np.timedelta64(45,'m');hi=np.datetime64(k['t_fin']);ins=(mt>=lo)&(mt<=hi)
        if ins.any(): hit|=ins;dethit+=1
    HH,CC=hxc(pd.read_csv(f"{d}/{subj}.csv",usecols=['Timestamp','Libre GL']).assign(T=lambda x:pd.to_datetime(x['Timestamp'])).set_index('T')['Libre GL'].resample('15min').mean().interpolate('time',limit_direction='both').values)
    rows.append(dict(subject=subj,grupo=gmap[sid],dias=ndays,H=round(HH,3),C=round(CC,3),
        comidas=len(ann),detecc=len(mk),detectadas=int(hit.sum()),det_ok=dethit,
        alba=len(dw),fuerte=lvl['fuerte'],moderada=lvl['moderada'],leve=lvl['leve']))
R=pd.DataFrame(rows);R.to_csv(_SAL+"/consolidado_3grupos.csv",index=False)
# agregado por grupo
agg=[]
for g in ['Sano','Prediabético','Diabético']:
    x=R[R.grupo==g]
    agg.append(dict(grupo=g,n=len(x),H=round(x.H.mean(),3),H_sd=round(x.H.std(),3),C=round(x.C.mean(),3),C_sd=round(x.C.std(),3),
        recall=round(x.detectadas.sum()/x.comidas.sum(),3),precision=round(x.det_ok.sum()/x.detecc.sum(),3),
        alba_dia=round(x.alba.sum()/x.dias.sum(),3)))
A=pd.DataFrame(agg);A.to_csv(_SAL+"/agg_3grupos.csv",index=False)
print(A.to_string(index=False))
