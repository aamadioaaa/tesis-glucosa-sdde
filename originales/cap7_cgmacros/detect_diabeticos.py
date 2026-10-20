from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
"""Diabéticos: ingestas (niveles) + fenómeno del alba (cap 3.2.6). Tasa de efectividad vs anotaciones."""
import pandas as pd, numpy as np, glob, os, pywt
STDIR=_SAL+"/S_t"; BASE=_CGM
DT_MIN=15; DT_H=DT_MIN/60; EPS=1e-9
DIAB=[3,5,12,14,23,28,29,30,35,36,38,39,41,42,45,46,47,49]
# ingestas
L_H=2.5; DGMIN=20.0; QMIN=0.2; BAND_FAST=(0.5,2.0); BAND_SLOW=(3.0,8.0); DAWN_DGMIN=30.0
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

def detect_meals(t,G,S,Wf,Ws,dawn_windows):
    """Detecta ingestas. Suprime toda ventana que solape una ventana de alba
    cuando la energía LENTA domina sobre la RÁPIDA (subida de alba, no comida)."""
    N=len(G); Lw=int(round(L_H/DT_H)); Wn=Wf/(Wf.max()+EPS)
    Wfn=Wf/(Wf.max()+EPS); Wsn=Ws/(Ws.max()+EPS)
    cands=[]
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
            # La ventana de ingesta NO puede empezar dentro de la franja del alba
            # (03:00-08:00) cuando ese día tiene evento de alba: se atribuye al alba.
            overlaps_dawn=any(not(tf<=e['t_ini'] or e['t_fin']<=ti) for e in dawn_windows)
            if overlaps_dawn and 3<=ti.hour<8:
                continue
            lab=classify(c['dg'],c['iG'],c['iS'],c['wmax'])
            if lab: c['clase']=lab; c['t_ini']=ti; c['t_fin']=tf; kept.append(c)
    return kept

def detect_dawn(t,G,S,Ws):
    """Ventana 03:00-08:00 por día; alba si ΔG>0, I_G>0, I_S>0 (banda lenta)."""
    df=pd.DataFrame({'t':t,'G':G,'S':S,'W':Ws/(Ws.max()+EPS)}); df['date']=df.t.dt.date
    df['mins']=df.t.dt.hour*60+df.t.dt.minute
    ev=[]
    for d,g in df.groupby('date'):
        w=g[(g.mins>=180)&(g.mins<=480)]   # 03:00–08:00 inclusive, corte exacto a las 8
        if len(w)<8: continue
        G0=w.G.iloc[0]; dg=w.G.iloc[-1]-G0
        iG=ipn((w.G-G0).values); iS=ipn(w.S.values); wmax=w.W.max()
        if dg>DAWN_DGMIN and iG>0 and iS>0:
            ev.append(dict(date=str(d),dG=round(dg,1),iG=round(iG,3),iS=round(iS,3),W=round(wmax,3),
                           t_ini=w.t.iloc[0],t_fin=w.t.iloc[-1]))
    return ev

allmeal=[]; alldawn=[]; summ=[]
tot_meal=tot_hit=tot_det=tot_det_hit=0; lvl_prec={'fuerte':[0,0],'moderada':[0,0],'leve':[0,0]}
for sid in DIAB:
    subj=f"CGMacros-{sid:03d}"
    st=pd.read_csv(f"{STDIR}/{subj}_St.csv",parse_dates=['Timestamp'])
    t=st['Timestamp']; G=st['G'].values; S=st['S'].values
    Wf=wenergy(S,BAND_FAST); Ws=wenergy(S,BAND_SLOW)
    dawn=detect_dawn(t,G,S,Ws)
    meals=detect_meals(t,G,S,Wf,Ws,dawn)
    ann=pd.read_csv(f"{BASE}/{subj}/{subj}.csv",usecols=['Timestamp','Meal Type']); ann['Timestamp']=pd.to_datetime(ann['Timestamp'])
    ann=ann[ann['Meal Type'].notna()]; mt=ann['Timestamp'].values
    meal_hit=np.zeros(len(ann),bool)
    for k in meals:
        lo=np.datetime64(k['t_ini'])-np.timedelta64(int(TOL_BEFORE_H*60),'m'); hi=np.datetime64(k['t_fin'])
        inside=(mt>=lo)&(mt<=hi); hit=inside.any()
        lvl_prec[k['clase']][0]+=1; lvl_prec[k['clase']][1]+=int(hit)
        if hit: meal_hit|=inside
        allmeal.append(dict(subject=subj,t_ini=k['t_ini'],t_fin=k['t_fin'],clase=k['clase'],
                            dG=round(k['dg'],1),iG=round(k['iG'],3),iS=round(k['iS'],3),Q=round(k['Q'],3),acierto=hit))
    for e in dawn: e2=dict(subject=subj,**e); alldawn.append(e2)
    nm=len(ann); nd=len(meals); nh=int(meal_hit.sum()); ndh=sum(1 for k in meals if True and any(
        (np.datetime64(k['t_ini'])-np.timedelta64(45,'m')<=mt)&(mt<=np.datetime64(k['t_fin']))))
    tot_meal+=nm; tot_hit+=nh; tot_det+=nd; tot_det_hit+=ndh
    summ.append(dict(subject=subj,comidas=nm,detecc=nd,detectadas=nh,alba_eventos=len(dawn),
                     recall=round(nh/nm,3) if nm else np.nan))
S1=pd.DataFrame(summ); S1.to_csv(_SAL+"/diab_resumen.csv",index=False)
pd.DataFrame(allmeal).to_csv(_SAL+"/diab_ingestas.csv",index=False)
pd.DataFrame(alldawn).to_csv(_SAL+"/diab_alba.csv",index=False)
print(S1.to_string(index=False))
print(f"\n===== DIABÉTICOS (n=18) =====")
print(f"RECALL ingestas: {tot_hit}/{tot_meal} = {100*tot_hit/tot_meal:.1f}%")
print(f"PRECISIÓN ingestas: {tot_det_hit}/{tot_det} = {100*tot_det_hit/tot_det:.1f}%")
print("\nPor nivel de detección (precisión = con comida real):")
for lv in ['fuerte','moderada','leve']:
    n,h=lvl_prec[lv]; print(f"  {lv:9s}: {h}/{n} = {100*h/n:.1f}%" if n else f"  {lv}: 0")
print(f"\nFenómeno del alba: {len(alldawn)} eventos en {sum(1 for s in summ if s['alba_eventos']>0)} sujetos")
