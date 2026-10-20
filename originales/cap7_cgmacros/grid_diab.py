from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Wedge
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
STDIR=_SAL+"/S_t"; BASE=_CGM
DIAB=[3,5,12,14,23,28,29,30,35,36,38,39,41,42,45,46,47,49]
BLUE='#1f6fdb'; LVLCOL={'fuerte':'#a80d00','moderada':'#e0342a','leve':'#f4907f'}
res=pd.read_csv(_SAL+"/diab_resumen.csv").set_index('subject')

def burger(color,s=1.0):
    W,H=26*s,20*s; da=DrawingArea(W,H,0,0); e='#3a2a15'
    da.add_artist(FancyBboxPatch((3,2),W-6,5,boxstyle="round,pad=0,rounding_size=2.5",fc=color,ec=e,lw=.8,mutation_aspect=0.6))
    da.add_artist(FancyBboxPatch((3.5,6),W-7,4,boxstyle="round,pad=0,rounding_size=1.8",fc='#6b3d1b',ec=e,lw=.6))
    da.add_artist(FancyBboxPatch((2.5,9),W-5,3,boxstyle="round,pad=0,rounding_size=2",fc='#7cb342',ec='#5a8a2c',lw=.5))
    da.add_artist(Wedge((W/2,12),(W-6)/2,0,180,width=(W-6)/2,fc=color,ec=e,lw=.8))
    for dx in (-6,-2.5,1,4.5): da.add_artist(Ellipse((W/2+dx,15.5+abs(dx)*.15),1.5,.9,fc='#fff4d6',ec='none'))
    return da

def rising_sun(s=1.0):
    W,H=24*s,14*s; da=DrawingArea(W,H,0,0); c='#f5a800'
    cx,cy,r=W/2,3,6.5*s
    # rayos
    import numpy as _np
    for ang in _np.linspace(10,170,7):
        a=_np.radians(ang); x0,y0=cx+ (r+1.5)*_np.cos(a), cy+(r+1.5)*_np.sin(a); x1,y1=cx+(r+5)*_np.cos(a),cy+(r+5)*_np.sin(a)
        da.add_artist(Line2D([x0,x1],[y0,y1],color=c,lw=1.6,solid_capstyle='round'))
    da.add_artist(Wedge((cx,cy),r,0,180,fc='#ffcb1f',ec='#e08a00',lw=1))
    return da

meals_cache={}
det=pd.read_csv(_SAL+"/diab_ingestas.csv",parse_dates=['t_ini','t_fin'])
alba=pd.read_csv(_SAL+"/diab_alba.csv",parse_dates=['t_ini','t_fin'])

fig,axes=plt.subplots(6,3,figsize=(22,26)); axes=axes.ravel()
plt.rcParams.update({'font.family':'DejaVu Sans'})
for i,sid in enumerate(DIAB):
    subj=f"CGMacros-{sid:03d}"; ax=axes[i]
    st=pd.read_csv(f"{STDIR}/{subj}_St.csv",parse_dates=['Timestamp'])
    t0=st.Timestamp.iloc[0].normalize()+pd.Timedelta(days=1); t1=t0+pd.Timedelta(days=2)
    g=st[(st.Timestamp>=t0)&(st.Timestamp<=t1)]
    m=pd.read_csv(f"{BASE}/{subj}/{subj}.csv",usecols=['Timestamp','Meal Type']); m['Timestamp']=pd.to_datetime(m['Timestamp']); m=m[m['Meal Type'].notna()]
    m=m[(m.Timestamp>=t0)&(m.Timestamp<=t1)]
    d=det[(det.subject==subj)&(det.t_ini>=t0)&(det.t_ini<=t1)]
    al=alba[(alba.subject==subj)&(alba.t_ini>=t0)&(alba.t_ini<=t1)]
    ax.plot(g.Timestamp,g.G,color='#2b2b2b',lw=1.3,zorder=3)
    ax.fill_between(g.Timestamp,g.G,g.G.min()-6,color='#2b2b2b',alpha=.04,zorder=1)
    for _,r in al.iterrows(): ax.axvspan(r.t_ini,r.t_fin,color='#f2c200',alpha=.25,zorder=0)
    for _,r in d.iterrows(): ax.axvspan(r.t_ini,r.t_fin,color=LVLCOL[r.clase],alpha=.08,zorder=0)
    gmax=g.G.max() if len(g) else 200; yb=gmax+34; yr=gmax+16
    for _,r in m.iterrows(): ax.add_artist(AnnotationBbox(burger(BLUE,.7),(r.Timestamp,yb),frameon=False,zorder=6))
    for _,r in d.iterrows(): ax.add_artist(AnnotationBbox(burger(LVLCOL[r.clase],.7),(r.t_ini,yr),frameon=False,zorder=6))
    for _,r in al.iterrows(): ax.add_artist(AnnotationBbox(rising_sun(.8),(r.t_ini+(r.t_fin-r.t_ini)/2,gmax+2),frameon=False,zorder=7))
    ax.set_ylim(g.G.min()-10,yb+18)
    rc=res.loc[subj,'recall'] if subj in res.index else np.nan
    ax.set_title(f"{subj}  (recall {rc:.0%}, {int(res.loc[subj,'alba_eventos'])} albas)",fontsize=11,weight='bold')
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.grid(axis='y',ls=':',alpha=.35)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)

# leyenda global
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=BLUE,markersize=11,label='Ingesta real'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor=LVLCOL['fuerte'],markersize=11,label='Detectada fuerte'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor=LVLCOL['moderada'],markersize=11,label='Detectada moderada'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor=LVLCOL['leve'],markersize=11,label='Detectada leve'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor='#ffcb1f',markersize=13,label='☀ Fenómeno del alba')]
fig.legend(handles=handles,loc='upper center',ncol=5,fontsize=13,bbox_to_anchor=(0.5,0.995),framealpha=.95)
fig.suptitle('Detección de ingestas y fenómeno del alba — 18 diabéticos (series desde 00:00, 2 días)',
             fontsize=17,weight='bold',y=1.008)
plt.tight_layout(rect=[0,0,1,0.985])
plt.savefig(_SAL+"/grid_diabeticos.png",dpi=120,bbox_inches='tight')
print("ok")
