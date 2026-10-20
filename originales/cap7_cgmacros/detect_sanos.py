from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
"""Corre la detección de ingestas+alba sobre los SANOS y compara con diabéticos."""
import pandas as pd, numpy as np, glob, os, pywt
STDIR=_SAL+"/S_t"; BASE=_CGM
DT_MIN=15; DT_H=DT_MIN/60; EPS=1e-9
L_H=2.5; DGMIN=20.0; QMIN=0.2; BAND_FAST=(0.5,2.0); BAND_SLOW=(3.0,8.0)
CLASS=[('fuerte',120,0.70,0.65,0.15),('moderada',70,0.55,0.35,0.0),('leve',25,0.0,0.0,0.0)]
TOL_BEFORE_H=0.75
def ipn(y):
    Ap=DT_H*np.sum(np.maximum(y,0)); An=DT_H*np.sum(np.maximum(-y,0)); return (Ap-An)/(Ap+An+EPS)
def wenergy(S,band):
    wav='cmor1.5-1.0'; per=np.linspace(band[0],band[1],12); fc=pywt.central_frequency(wav)
    scales=fc/((1/per)*DT_H); coef,_=pywt.cwt(S,scales,wav,sampling_period=DT_H)
    return np.mean(np.abs(coef)**2,axis=0)
def classify(dg,iG,iS,wmax):
    for name,dgt,igt,ist,wt in CLASS:
        if dg>=dgt and iG>igt and iS>ist and wmax>wt: return name
    return None
def detect_dawn(t,G,S,Ws):
    df=pd.DataFrame({'t':t,'G':G,'S':S,'W':Ws/(Ws.max()+EPS)}); df['date']=df.t.dt.date; ev=[]
    df['mins']=df.t.dt.hour*60+df.t.dt.minute
    for d,g in df.groupby('date'):
        w=g[(g.mins>=180)&(g.mins<=480)]   # 03:00–08:00 inclusive
        if len(w)<8: continue
        G0=w.G.iloc[0]; dg=w.G.iloc[-1]-G0; iG=ipn((w.G-G0).values); iS=ipn(w.S.values); wmax=w.W.max()
        if dg>30.0 and iG>0 and iS>0:
            ev.append(dict(date=str(d),dG=round(dg,1),iG=round(iG,3),iS=round(iS,3),W=round(wmax,3),t_ini=w.t.iloc[0],t_fin=w.t.iloc[-1]))
    return ev
def detect_meals(t,G,S,Wf,dawn):
    N=len(G); Lw=int(round(L_H/DT_H)); Wn=Wf/(Wf.max()+EPS); cands=[]
    for a in range(0,N-Lw):
        b=a+Lw; Gw=G[a:b]; Sw=S[a:b]; dg=Gw.max()-Gw[0]
        if dg<DGMIN: continue
        iG=ipn(Gw-Gw[0]); iS=ipn(Sw); wmax=Wn[a:b].max()
        Q=(dg/40)*(max(iS,0)+EPS)*(1+min(wmax,1.5))*(max(iG,0)+EPS)
        if Q<QMIN: continue
        cands.append(dict(a=a,b=b,dg=dg,iG=iG,iS=iS,wmax=wmax,Q=Q))
    cands.sort(key=lambda c:-c['Q']); kept=[]
    for c in cands:
        if all(c['b']<=k['a'] or k['b']<=c['a'] for k in kept):
            ti=t.iloc[c['a']]; tf=t.iloc[min(c['b'],N-1)]
            od=any(not(tf<=e['t_ini'] or e['t_fin']<=ti) for e in dawn)
            if od and 3<=ti.hour<8: continue
            lab=classify(c['dg'],c['iG'],c['iS'],c['wmax'])
            if lab: c['clase']=lab; c['t_ini']=ti; c['t_fin']=tf; kept.append(c)
    return kept

# grupos
b=pd.read_csv(f"{BASE}/bio.csv"); b['A1c']=pd.to_numeric(b['A1c PDL (Lab)'],errors='coerce'); b['F']=pd.to_numeric(b['Fasting GLU - PDL (Lab)'],errors='coerce')
def cl(r):
    a,f=r['A1c'],r['F']
    if (pd.notna(a) and a>=6.5) or (pd.notna(f) and f>=126): return 'Diab'
    if (pd.notna(a) and a>=5.7) or (pd.notna(f) and f>=100): return 'Pre'
    return 'Sano'
b['g']=b.apply(cl,axis=1)
SAN=sorted(int(s) for s in b[b.g=='Sano']['subject'])

rows=[]; tot_days=0; tot_dawn=0; tot_meal=tot_hit=0
for sid in SAN:
    subj=f"CGMacros-{sid:03d}"
    if not os.path.exists(f"{STDIR}/{subj}_St.csv"): continue
    st=pd.read_csv(f"{STDIR}/{subj}_St.csv",parse_dates=['Timestamp'])
    t=st['Timestamp']; G=st['G'].values; S=st['S'].values
    Wf=wenergy(S,BAND_FAST); Ws=wenergy(S,BAND_SLOW)
    dawn=detect_dawn(t,G,S,Ws); meals=detect_meals(t,G,S,Wf,dawn)
    ndays=t.dt.date.nunique()
    ann=pd.read_csv(f"{BASE}/{subj}/{subj}.csv",usecols=['Timestamp','Meal Type']); ann['Timestamp']=pd.to_datetime(ann['Timestamp']); ann=ann[ann['Meal Type'].notna()]; mt=ann['Timestamp'].values
    hit=np.zeros(len(ann),bool)
    for k in meals:
        lo=np.datetime64(k['t_ini'])-np.timedelta64(45,'m'); hi=np.datetime64(k['t_fin']); ins=(mt>=lo)&(mt<=hi)
        if ins.any(): hit|=ins
    rows.append(dict(subject=subj,dias=ndays,alba=len(dawn),alba_por_dia=round(len(dawn)/ndays,2),
                     comidas=len(ann),detecc=len(meals),recall=round(hit.mean(),3) if len(ann) else np.nan))
    tot_days+=ndays; tot_dawn+=len(dawn); tot_meal+=len(ann); tot_hit+=int(hit.sum())
R=pd.DataFrame(rows); R.to_csv(_SAL+"/sanos_verificacion.csv",index=False)
print("SANOS (n=%d):"%len(rows)); print(R.to_string(index=False))
print(f"\nAlba SANOS: {tot_dawn} eventos en {tot_days} días = {tot_dawn/tot_days:.2f}/día")
print(f"Recall ingestas SANOS: {tot_hit}/{tot_meal} = {100*tot_hit/tot_meal:.1f}%")
# comparativo diabéticos
dd=pd.read_csv(_SAL+"/diab_resumen.csv")
# dias diabéticos
ddias=0
for subj in dd.subject:
    st=pd.read_csv(f"{STDIR}/{subj}_St.csv",parse_dates=['Timestamp']); ddias+=st['Timestamp'].dt.date.nunique()
print(f"\nComparativo alba/día:  SANOS {tot_dawn/tot_days:.2f}   vs   DIABÉTICOS {dd.alba_eventos.sum()/ddias:.2f}")
