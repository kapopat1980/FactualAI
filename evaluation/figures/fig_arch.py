import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8})
fig,ax=plt.subplots(figsize=(6.6,5.4),dpi=300); ax.set_xlim(0,100); ax.set_ylim(-2,100); ax.axis("off")
def box(x,y,w,h,title,sub="",fc="#ffffff",ec="#333",bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=0.8))
    ax.text(x+w/2,y+h/2+(1.6 if sub else 0),title,ha="center",va="center",fontsize=7.6,weight="bold" if bold else "normal")
    if sub: ax.text(x+w/2,y+h/2-2.2,sub,ha="center",va="center",fontsize=6.3,color="#333")
def arr(x1,y1,x2,y2,ls="-",c="#333"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.8,color=c,linestyle=ls,shrinkA=0,shrinkB=0))
L="#e8f0fb"; O="#fdeee6"; G="#f2f2f2"
# inputs
ins=[("Text","any language"),("Image","OpenCV +\nTesseract OCR"),("Audio file","Faster-Whisper\nASR"),("Video URL","yt-dlp → FFmpeg\n→ ASR"),("RSS feed","feedparser,\n4 outlets")]
xs=[1,21,41,61,81]
for (t,s),x in zip(ins,xs): box(x,87,17,9,t,s,fc=G)
ax.text(50,99,"Input layer (React.js client → Flask REST API)",ha="center",fontsize=7.5,style="italic")
for x in xs[:4]: arr(x+8.5,87,50,79.5)
box(22,70,56,9,"Language identification & translation","LangDetect / Whisper LID → Google Translate → English",fc=L)
box(22,58,56,8,"Normalisation","lower-casing, punctuation & whitespace removal",fc=L)
arr(50,70,50,66.5)
# RSS bypass
ax.annotate("",xy=(90,50),xytext=(90,87),arrowprops=dict(arrowstyle="-|>",lw=0.8,color="#777",linestyle="--"))
ax.text(94.5,66,"RF only\n(no LLM)",fontsize=6.2,color="#555",ha="center")
# two branches
arr(42,58,22,50); arr(58,58,78,50)
box(60,38,36,11.5,"Random Forest (200 trees)","all-mpnet-base-v2 embedding (768-d)\n→ RF_fake = P(fake)",fc=L)
box(4,38,36,11.5,"LLaMA 3.2 1B via Ollama","first 1,500 chars, JSON prompt\n→ label, confidence, justification",fc=O)
# fallback note
box(4,24,36,9,"LLM fallback","timeout (30 s), parse error\nor “Unknown” ⇒ LLM_fake := RF_fake",fc="#ffffff",ec="#999",bold=True)
arr(22,38,22,33.5,ls="--",c="#777")
arr(78,38,58,20.5); arr(22,24,42,20.5); arr(34,38,44,20.5)
box(20,11,60,9,"Weighted ensemble","ensemble_fake = w·RF_fake + (1 − w)·LLM_fake   (deployed w = 0.6)",fc=G)
arr(50,11,50,7.4)
box(8,-1,84,8,"Output (JSON / UI)","label · confidence · per-model scores · source language\nLLM justification · transcript / OCR text",fc=G)
plt.savefig("evaluation/figures/fig_arch.png",bbox_inches="tight",pad_inches=0.05)
