"""GS6 step 3 (GPU on A6000; CPU only for a smoke test): TotalSegmentator full-resolution organ model (Dataset291 part1
organs, 1.5 mm, non-fast) on the 1.5 mm raw crops of gs6_rawcrop.py, with the predictor loaded once (same settings as
src/worker.py --engine nnunet: tile step 0.5, gaussian, no mirroring; baseline CTNormalization -> sliding window -> argmax).
Writes out_dir/{id}.nii.gz uint8 (1 = gallbladder, 2 = liver) at 1.5 mm with the crop affine, and seg_log.csv.
Polls the inbox until <inbox>/DONE_SENDING exists and every file is done; deletes each input after success if --consume.
usage: gs6_seg.py --inbox DIR --out DIR [--device cuda] [--consume] [--limit N]"""
import argparse, os, sys, time, csv, numpy as np, nibabel as nib, torch
from pathlib import Path
SRC = os.environ.get("MW_SRC", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src"); sys.path.insert(0, SRC)
from encoder import NnUNetEncoder
from windowing import apply_window
MD = os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres")
ap = argparse.ArgumentParser(); ap.add_argument("--inbox", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda"); ap.add_argument("--consume", action="store_true"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(2)
enc = NnUNetEncoder(MD, device=a.device)
GB, LIV = int(enc.labels["gallbladder"]), int(enc.labels["liver"])
os.makedirs(a.out, exist_ok=True)
logf = Path(a.out) / "seg_log.csv"; new = not logf.exists()
fh = open(logf, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["study_id", "shape", "gb_vox15", "liver_vox15", "seconds"])


@torch.no_grad()
def seg(hu):
    x = torch.from_numpy(apply_window(hu, {"type": "baseline"}, enc.norm)).float()[None]
    try:
        lg = enc.predictor.predict_sliding_window_return_logits(x)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); enc.predictor.perform_everything_on_device = False
        try: lg = enc.predictor.predict_sliding_window_return_logits(x)
        finally: enc.predictor.perform_everything_on_device = True
    lab = lg.argmax(0).to(torch.uint8).cpu().numpy(); del lg
    return lab


n, t0 = 0, time.time(); failed = set()
while True:
    todo = sorted(p for p in Path(a.inbox).glob("AC*.nii.gz") if not p.name.endswith(".tmp.nii.gz")
                  and not (Path(a.out) / p.name).exists() and p.name not in failed)
    if not todo:
        if (Path(a.inbox) / "DONE_SENDING").exists() or (Path(a.inbox) / "STOP").exists(): break
        time.sleep(30); continue
    for p in todo:
        if (Path(a.inbox) / "STOP").exists(): break
        t = time.time(); sid = p.name.split(".")[0]
        try:
            img = nib.load(str(p)); hu = np.asanyarray(img.dataobj).astype(np.int16).transpose(2, 1, 0)  # already canonical 1.5 mm
            lab = seg(hu)
            out = np.zeros_like(lab); out[lab == GB] = 1; out[lab == LIV] = 2
            tmp = Path(a.out) / f"{sid}.tmp.nii.gz"
            nib.save(nib.Nifti1Image(out.transpose(2, 1, 0), img.affine), str(tmp)); tmp.rename(Path(a.out) / f"{sid}.nii.gz")
            wr.writerow([sid, "x".join(map(str, hu.shape)), int((out == 1).sum()), int((out == 2).sum()), f"{time.time()-t:.2f}"]); fh.flush()
            if a.consume: p.unlink()
            n += 1
            if n % 100 == 0: print(f"{n} cases {time.time()-t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(sid, "ERROR", repr(e)[:200], flush=True); failed.add(p.name)
        torch.cuda.empty_cache()
        if a.limit and n >= a.limit: break
    if a.limit and n >= a.limit: break
print("GS6_SEG_DONE", n, f"{time.time()-t0:.0f}s", flush=True)
