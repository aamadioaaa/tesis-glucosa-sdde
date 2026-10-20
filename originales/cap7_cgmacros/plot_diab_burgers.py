from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Wedge
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
import matplotlib.dates as mdates
STDIR=_SAL+"/S_t"; BASE=_CGM
SUBJ="CGMacros-029"
BLUE='#1f6fdb'; LVLCOL={'fuerte':'#a80d00','moderada':'#e0342a','leve':'#f4907f'}

def burger(color,s=1.0):
    W,H=26*s,20*s; da=DrawingArea(W,H,0,0); edge='#3a2a15'
    da.add_artist(FancyBboxPatch((3,2),W-6,5,boxstyle="round,pad=0,rounding_size=2.5",fc=color,ec=edge,lw=1,mutation_aspect=0.6))
    da.add_artist(FancyBboxPatch((3.5,6),W-7,4,boxstyle="round,pad=0,rounding_size=1.8",fc='#6b3d1b',ec=edge,lw=.8))
    da.add_artist(FancyBboxPatch((2.5,9),W-5,3,boxstyle="round,pad=0,rounding_size=2",fc='#7cb342',ec='#5a8a2c',lw=.6))
    da.add_artist(Wedge((W/2,12),(W-6)/2,0,180,width=(W-6)/2,fc=color,ec=edge,lw=1))
    for dx in (-6,-2.5,1,4.5): da.add_artist(Ellipse((W/2+dx,15.5+abs(dx)*0.15),1.6,1.0,fc='#fff4d6',ec='none'))
    return da

st=pd.read_csv(f"{STDIR}/{SUBJ}_St.csv",parse_dates=['Timestamp'])
det=pd.read_csv(_SAL+"/diab_ingestas.csv",parse_dates=['t_ini','t_fin']); det=det[det.subject==SUBJ]
alba=pd.read_csv(_SAL+"/diab_alba.csv",parse_dates=['t_ini','t_fin']); alba=alba[alba.subject==SUBJ]
meals=pd.read_csv(f"{BASE}/{SUBJ}/{SUBJ}.csv",usecols=['Timestamp','Meal Type']); meals['Timestamp']=pd.to_datetime(meals['Timestamp']); meals=meals[meals['Meal Type'].notna()]
# alinear desde primera medianoche, 3.5 días
t0=st.Timestamp.iloc[0].normalize()+pd.Timedelta(days=1); t1=t0+pd.Timedelta(days=3.5)
g=st[(st.Timestamp>=t0)&(st.Timestamp<=t1)]
meals=meals[(meals.Timestamp>=t0)&(meals.Timestamp<=t1)]; det=det[(det.t_ini>=t0)&(det.t_ini<=t1)]; alba=alba[(alba.t_ini>=t0)&(alba.t_ini<=t1)]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,ax=plt.subplots(figsize=(15,6.4))
ax.plot(g.Timestamp,g.G,color='#2b2b2b',lw=1.6,zorder=3)
ax.fill_between(g.Timestamp,g.G,g.G.min()-8,color='#2b2b2b',alpha=0.04,zorder=1)
# ventanas alba (banda dorada)
for _,r in alba.iterrows():
    ax.axvspan(r.t_ini,r.t_fin,color='#f2c200',alpha=0.28,zorder=0)
    ax.text(r.t_ini+(r.t_fin-r.t_ini)/2,g.G.max()+2,'☀ alba',ha='center',fontsize=9,color='#9a7b00')
# ventanas ingesta detectadas (rojo claro)
for _,r in det.iterrows():
    ax.axvspan(r.t_ini,r.t_fin,color=LVLCOL[r.clase],alpha=0.08,zorder=0)
gmax=g.G.max(); y_blue=gmax+30; y_red=gmax+14
for _,r in meals.iterrows():
    ax.add_artist(AnnotationBbox(burger(BLUE),(r.Timestamp,y_blue),frameon=False,zorder=6,box_alignment=(0.5,0.5)))
for _,r in det.iterrows():
    ax.add_artist(AnnotationBbox(burger(LVLCOL[r.clase]),(r.t_ini,y_red),frameon=False,zorder=6,box_alignment=(0.5,0.5)))
ax.set_ylim(g.G.min()-12,y_blue+22); ax.set_ylabel('Glucosa  G(t)  [mg/dL]',fontsize=12)
ax.set_title(f'{SUBJ} (diabético) — Ingestas anotadas vs. detectadas + fenómeno del alba',fontsize=13.5,weight='bold',pad=12)
ax.grid(axis='y',ls=':',alpha=.4)
ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6,12,18]))
for s in ['top','right']: ax.spines[s].set_visible(False)
# leyenda
lg=fig.add_axes([0.085,0.14,0.30,0.20]); lg.set_xlim(0,10); lg.set_ylim(0,4); lg.set_xticks([]); lg.set_yticks([])
lg.set_facecolor('#fbfbfb'); [sp.set_edgecolor('#ccc') for sp in lg.spines.values()]
lg.add_artist(AnnotationBbox(burger(BLUE,.8),(0.8,3.4),frameon=False)); lg.text(1.7,3.4,'Ingesta real (tabla)',va='center')
lg.add_artist(AnnotationBbox(burger(LVLCOL['fuerte'],.8),(0.8,2.5),frameon=False)); lg.text(1.7,2.5,'Detectada — fuerte',va='center')
lg.add_artist(AnnotationBbox(burger(LVLCOL['moderada'],.8),(0.8,1.6),frameon=False)); lg.text(1.7,1.6,'Detectada — moderada',va='center')
lg.add_artist(AnnotationBbox(burger(LVLCOL['leve'],.8),(0.8,0.7),frameon=False)); lg.text(1.7,0.7,'Detectada — leve',va='center')
plt.savefig(_SAL+"/diab_hamburguesas_029.png",dpi=140)
print("ok",len(meals),"reales,",len(det),"detectadas,",len(alba),"alba")
