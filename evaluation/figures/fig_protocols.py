import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8.5})
cv=json.load(open("evaluation/results/results_cv.json")); ex=json.load(open("evaluation/results/results_extra.json"))
labs=["P0\nOriginal","P1\nDe-dup.","P2\nScrubbed","P2\nTitles","P3\nTemporal","P4\n→PolitiFact","P4\n→GossipCop"]
def series(k):
    return [cv["P0"][k]["acc"][0],cv["P1"][k]["acc"][0],cv["P2"][k]["acc"][0],cv["P2_title"][k]["acc"][0],ex["temporal"][k]["acc"],ex["cross"][f"{k} -> politifact"]["acc"],ex["cross"][f"{k} -> gossipcop"]["acc"]]
fig,ax=plt.subplots(figsize=(6.5,3.2),dpi=300)
x=range(len(labs))
A=[100*v for v in series("TF-IDF + LR")]; B=[100*v for v in series("RF on LSA-768")]
for k,c,m,ls,name in [("TF-IDF + LR","#2a78d6","o","-","TF-IDF + LR"),("RF on LSA-768","#eb6834","s","--","RF on LSA-768 (dense analogue)")]:
    y=A if k=="TF-IDF + LR" else B; other=B if k=="TF-IDF + LR" else A
    ax.plot(x,y,color=c,marker=m,ms=5,lw=1.6,ls=ls,label=name,mec="white",mew=0.8)
    for xi,yi in zip(x,y):
        off=6 if yi>=other[xi] else -11
        ax.annotate(f"{yi:.1f}",(xi,yi),textcoords="offset points",xytext=(0,off),ha="center",fontsize=6.5,color="#333")
pf=ex["fnn_counts"]["politifact"]; gc=ex["fnn_counts"]["gossipcop"]
for xi,n in [(5,pf),(6,gc)]:
    maj=100*max(n.values())/sum(n.values())
    ax.hlines(maj,xi-0.3,xi+0.3,color="#555",lw=1.2,ls=":")
    ax.text(xi-0.34,maj,f"majority\n{maj:.1f}",fontsize=6,va="center",ha="right",color="#555")
ax.axvline(3.5,color="#bbb",lw=0.6); ax.text(1.5,30,"within ISOT (5-fold CV)",ha="center",fontsize=7,color="#555"); ax.text(5,30,"distribution shift",ha="center",fontsize=7,color="#555")
ax.set_xticks(list(x)); ax.set_xticklabels(labs,fontsize=7.5); ax.set_ylim(25,104); ax.set_ylabel("Accuracy (%)")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y",color="#e5e5e5",lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False,loc="lower left",fontsize=7.5,bbox_to_anchor=(0,0.08))
plt.tight_layout(); plt.savefig("evaluation/figures/fig_protocols.png")
