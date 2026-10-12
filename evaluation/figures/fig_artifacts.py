import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9})
r=pd.read_csv("evaluation/results/artifacts.csv")
lab={'(Reuters) tag':'“(Reuters)” tag in body','Dateline "CITY (Reuters) -"':'Dateline “CITY (Reuters) –”','"Featured image"':'“Featured image” credit',
'"21st Century Wire"':'“21st Century Wire” brand','URL (http/www)':'URL in body','pic.twitter.com':'Embedded pic.twitter.com link','@handle':'@-handle in body',
'"Getty Images"':'“Getty Images” credit','"via" photo credit':'“image via / photo by” credit','Title fully upper-case word (>=1)':'ALL-CAPS word (≥4 letters) in title'}
r["m"]=r.marker.map(lab); r["gap"]=(r.fake_pct-r.real_pct).abs(); r=r.sort_values("gap")
fig,ax=plt.subplots(figsize=(6.5,3.6),dpi=300)
y=range(len(r)); h=0.38
ax.barh([i+h/2 for i in y],r.fake_pct,h,color="#2a78d6",label="Fake (n = 23,481)",edgecolor="white",linewidth=0.6)
ax.barh([i-h/2 for i in y],r.real_pct,h,color="#eb6834",label="Real (n = 21,417)",hatch="////",edgecolor="white",linewidth=0.6)
for i,(f,t) in enumerate(zip(r.fake_pct,r.real_pct)):
    ax.text(f+1,i+h/2,f"{f:.1f}",va="center",fontsize=7,color="#333")
    ax.text(t+1,i-h/2,f"{t:.1f}",va="center",fontsize=7,color="#333")
ax.set_yticks(list(y)); ax.set_yticklabels(r.m); ax.set_xlim(0,108); ax.set_xlabel("Articles containing the marker (%)")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x",color="#ddd",linewidth=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False,loc="lower right",fontsize=8)
plt.tight_layout(); plt.savefig("evaluation/figures/fig_artifacts.png")
