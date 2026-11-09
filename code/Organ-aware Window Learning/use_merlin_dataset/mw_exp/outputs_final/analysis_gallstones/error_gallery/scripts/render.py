import sys, numpy as np, pandas as pd, matplotlib, json, html
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont
D, OUT = sys.argv[1], sys.argv[2]
FONT="/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(FONT); fam=fm.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"]=fam; plt.rcParams["axes.unicode_minus"]=False
Z=np.load(f"{D}/slices.npz"); S=pd.read_csv(f"{D}/sel_anon.csv")
thr_p=1/(1+np.exp(-S.thr.iloc[0]))
CAT={"FN-F":"漏診｜F 非高密度結石（3 mm 下難以看見）","FN-G":"漏診｜G 高密度結石可見，模型仍漏掉",
"FN-B":"漏診｜B 膽囊遮罩缺失／過小（<100 vox）","FN-D":"漏診｜D 報告用語不確定（probably／possible）",
"FP-b200":"誤報｜b 膽囊內大量 >200 HU（疑似漏標或顯影劑）","FP-bPV":"誤報｜b 膽囊內少量 >130 HU（部分容積／壁或鄰近強化）",
"FP-chole":"誤報｜膽囊切除術後，但膽囊遮罩仍有體素（疑誤分割）","FP-f":"誤報｜f 其他（膽囊內無明顯高密度）",
"TP":"正確偵測（TP）｜明顯鈣化結石","TN":"正確陰性（TN）｜膽囊內無高密度"}
GRP={"FN":"FN","FP":"FP","TP":"TP","TN":"TN"}
cnt={}; S["anon"]=""
for i,r in S.iterrows():
    g=r.tag.split("-")[0]; cnt[g]=cnt.get(g,0)+1; S.loc[i,"anon"]=f"{g}-{cnt[g]}"
def win(a,L=40,W=400): return np.clip((a-(L-W/2))/W,0,1)
def disp(a): return a[::-1, ::-1]   # rows: anterior/superior on top; cols: patient right on image left
HALF=22
recs=[]
for _,r in S.iterrows():
    k=r.key; g=lambda n: Z[f"{k}|{n}"]
    fig,ax=plt.subplots(2,3,figsize=(12,8.9),dpi=120,gridspec_kw=dict(hspace=0.12,wspace=0.04,top=0.80,bottom=0.01,left=0.01,right=0.99))
    panels=[("axA","gbA","cA",f"軸切面：膽囊面積最大層"),("axM","gbM","cM","軸切面：膽囊內最高 HU 層"),("cor","gbC","cC","冠狀切面：通過膽囊中心")]
    for j,(im,mk,c,title) in enumerate(panels):
        a=disp(g(im).astype(float)); m=disp(g(mk).astype(float)); H,W=a.shape
        cy,cx=g(c); cy,cx=H-1-cy,W-1-cx
        for row in (0,1):
            A=ax[row,j]; A.set_facecolor("black"); A.imshow(win(a),cmap="gray",vmin=0,vmax=1,interpolation="nearest" if row else "bilinear")
            if m.sum()>0: A.contour(m,levels=[0.5],colors=["#39ff14"],linewidths=0.9 if row==0 else 1.4)
            if row==0:
                A.add_patch(plt.Rectangle((cx-HALF,cy-HALF),2*HALF,2*HALF,fill=False,ec="#ffd400",lw=1,ls="--"))
                A.set_title(title,fontsize=12); A.set_xlim(-0.5,W-0.5); A.set_ylim(H-0.5,-0.5)
            else:
                A.set_xlim(cx-HALF,cx+HALF); A.set_ylim(cy+HALF,cy-HALF)
                if im=="axM" and m.sum()>0:
                    A.plot([cx-5,cx-2],[cy,cy],c="#ff3b3b",lw=1.2); A.plot([cx+2,cx+5],[cy,cy],c="#ff3b3b",lw=1.2)
                    A.plot([cx,cx],[cy-5,cy-2],c="#ff3b3b",lw=1.2); A.plot([cx,cx],[cy+2,cy+5],c="#ff3b3b",lw=1.2)
                A.text(1,1,"放大（約 13 cm 視野）" if row else "",transform=A.transAxes,ha="right",va="top",color="w",fontsize=9)
            A.set_xticks([]); A.set_yticks([])
            A.text(0.01,0.01,"R",transform=A.transAxes,color="#ffd400",fontsize=10)
    vol=r.gb_vox*27/1000
    lab="膽結石 陽性" if r.y==1 else "膽結石 陰性"
    if r.y_chole==1: lab+="（另標：膽囊切除術後）"
    gbmax="—" if pd.isna(r.gb_max) else f"{r.gb_max:.0f}"
    pred="判為陽性" if r.p>=thr_p else "判為陰性"
    cap1=f"{r.anon}　{CAT[r.tag]}"
    cap2=f"模型分數 {r.p:.3f}（閾值 {thr_p:.3f}，90% 特異度）→ {pred}　｜　標註：{lab}"
    cap3=f"膽囊體積 {vol:.1f} mL（{int(r.gb_vox)} vox）　｜　膽囊最大 HU {gbmax}　｜　>130 HU 體素 {int(r.n130) if not pd.isna(r.n130) else 0}　｜　>200 HU 體素 {int(r.n200) if not pd.isna(r.n200) else 0}"
    fig.text(0.01,0.965,cap1,fontsize=16,weight="bold",va="top")
    fig.text(0.01,0.915,cap2,fontsize=12,va="top"); fig.text(0.01,0.875,cap3,fontsize=12,va="top")
    fig.text(0.99,0.965,"腹部窗 L40／W400　綠線＝膽囊遮罩　紅十字＝最高 HU 位置",fontsize=9,ha="right",va="top",color="#555")
    fn=f"{r.anon}.png"; fig.savefig(f"{OUT}/{fn}",facecolor="white"); plt.close(fig)
    recs.append(dict(file=fn,anon=r.anon,cap1=cap1,cap2=cap2,cap3=cap3,tag=r.tag))
json.dump(recs,open(f"{D}/recs.json","w"),ensure_ascii=False)
# montage: 3 columns
ims=[Image.open(f"{OUT}/{x['file']}").convert("RGB") for x in recs]
w=1300; ims=[im.resize((w,int(im.height*w/im.width)),Image.LANCZOS) for im in ims]
h=ims[0].height; cols=3; rows=-(-len(ims)//cols); pad=20; head=110
M=Image.new("RGB",(cols*w+(cols+1)*pad,head+rows*(h+pad)+pad),"white")
dr=ImageDraw.Draw(M); f=ImageFont.truetype(FONT,52); f2=ImageFont.truetype(FONT,28)
dr.text((pad,18),"膽結石模型錯誤圖庫（train 5-fold OOF，E3_late，90% 特異度）",font=f,fill="black")
dr.text((pad,78),"FN＝漏診 6 例｜FP＝誤報 4 例｜TP／TN＝對照各 2 例｜只用 train，未使用 val／test",font=f2,fill="#444")
for i,im in enumerate(ims):
    x=pad+(i%cols)*(w+pad); y=head+pad+(i//cols)*(h+pad); M.paste(im,(x,y))
M.save(f"{OUT}/montage_overview.png",optimize=True); print("montage",M.size)
