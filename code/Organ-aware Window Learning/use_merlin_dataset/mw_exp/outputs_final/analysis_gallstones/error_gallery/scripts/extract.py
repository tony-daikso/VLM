import numpy as np, pandas as pd, os
from scipy import ndimage
P=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs/prep/")
S=pd.read_csv(os.path.expanduser("~/gs_gallery_tmp/sel.csv"))
out={}
cnt={}
for _,r in S.iterrows():
    cnt[r.tag]=cnt.get(r.tag,0)+1; key=f"{r.tag}_{cnt[r.tag]}"
    z=np.load(P+r.study_id+".npz"); names=list(z["unit_names"]); hu=z["hu"]; u=z["units"]
    gb=u==names.index("gallbladder")+1; lv=u==names.index("liver")+1
    sp=np.abs(np.diag(z["affine"])[:3])
    if gb.sum()>=5:
        area=gb.sum((1,2)); zA=int(area.argmax())
        h=np.where(gb,hu,-3000); zM,yM,xM=np.unravel_index(h.argmax(),h.shape)
        cz,cy,cx=ndimage.center_of_mass(gb); src="gb"
    else:
        # fossa fallback: inferior-medial liver: lowest 25% liver slices, medial half
        zz,yy,xx=np.nonzero(lv); zt=np.percentile(zz,25); m=zz<=zt
        med=np.median(xx[m]); m2=m&(xx<=med) # medial = toward patient left = smaller x in RAS
        cz,cy,cx=zz[m2].mean(),yy[m2].mean(),xx[m2].mean(); zA=zM=int(round(cz)); yM,xM=int(cy),int(cx); src="fossa"
    yc=int(round(cy))
    out[key]=dict(axA=hu[zA],gbA=gb[zA],axM=hu[zM],gbM=gb[zM],cor=hu[:,yc,:],gbC=gb[:,yc,:],
                  cA=np.array([cy,cx]),cM=np.array([yM,xM]),cC=np.array([cz,cx]),sp=sp,src=src,zA=zA,zM=zM,
                  lvx=np.array(ndimage.center_of_mass(lv)) if lv.sum() else np.array([np.nan]*3))
    print(key,src,hu.shape,sp,zA,zM,"liver com",out[key]["lvx"],"gb com",(cz,cy,cx))
flat={}
for k,d in out.items():
    for kk,v in d.items(): flat[f"{k}|{kk}"]=np.asarray(v)
np.savez_compressed(os.path.expanduser("~/gs_gallery_tmp/slices.npz"),**flat)
S.assign(key=[f"{t}_{i}" for t,i in zip(S.tag,S.groupby("tag").cumcount()+1)]).drop(columns="study_id").to_csv(os.path.expanduser("~/gs_gallery_tmp/sel_anon.csv"),index=False)
