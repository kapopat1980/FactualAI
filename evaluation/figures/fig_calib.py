import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8.5})
d=json.load(open("evaluation/results/results_calib_lodo.json"))
fig,(a,b)=plt.subplots(1,2,figsize=(6.6,3.4),dpi=300)
a.plot([0,1],[0,1],color="#999",lw=0.8,ls=":")
for k,c,m,ls,lab in [("rf_mpnet_holdout","#2a78d6","o","-","In-distribution (P2 split)"),("rf_mpnet_temporal","#eb6834","s","--","Temporal shift (P3)")]:
    r=d[k]["reliability"]; a.plot([x[0] for x in r],[x[1] for x in r],color=c,marker=m,ms=4,lw=1.5,ls=ls,label=lab,mec="white",mew=0.6)
a.set_xlabel("Predicted P(fake)"); a.set_ylabel("Observed share fake"); a.set_title("(a) Reliability, production classifier",fontsize=8.5)
a.set_xlim(0,1); a.set_ylim(0,1); a.legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=1)
for k,c,m,ls,lab in [("rf_mpnet_holdout","#2a78d6","o","-","RF-mpnet, P2"),("rf_mpnet_temporal","#eb6834","s","--","RF-mpnet, P3"),("rf_mpnet_temporal_isotonic","#eb6834","^",":","RF-mpnet, P3, isotonic"),("lr_temporal","#555555","D","-.","TF-IDF + LR, P3")]:
    cv=d[k]["coverage"]; b.plot([100*x[0] for x in cv],[100*x[1] for x in cv],color=c,marker=m,ms=4,lw=1.4,ls=ls,label=lab,mec="white",mew=0.6)
b.invert_xaxis(); b.set_xlabel("Coverage (%) – share of inputs answered"); b.set_ylabel("Accuracy on answered inputs (%)"); b.set_title("(b) Selective prediction (abstention)",fontsize=8.5)
b.legend(frameon=False,fontsize=6.8,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2)
for ax in (a,b):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(color="#e5e5e5",lw=0.5); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("evaluation/figures/fig_calib.png")
