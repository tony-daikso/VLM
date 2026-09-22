"""
Run Merlin zero-shot contrastive inference over the 72 abdomen CT NIfTI volumes,
scored against the same 90 finding prompts used for the Pillar0 comparison.
"""
import csv
import json
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from merlin import Merlin
from merlin.data import DataLoader

NIFTI_DIR = "/datadrive/VLM/Merlin/CT/nifti"
CACHE_DIR = "/datadrive/VLM/Merlin/CT/cache"
FINDING_MAP_PATH = "/datadrive/VLM/Pillar0/CT/finding_english_map.json"
OUT_CSV = "/root/Desktop/VLM/Merlin/CT/results/Merlin_infer_results_CG2_abdomen72_v2_reportstyle.csv"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

finding_map = json.load(open(FINDING_MAP_PATH, encoding="utf-8"))
keys = list(finding_map.keys())
# Report-section style, matching Merlin's training report format (e.g. "Liver and biliary tree: Normal.")
prompts = [f"{finding_map[k]['organ_en']}: {finding_map[k]['finding_en']}." for k in keys]
print(f"{len(prompts)} finding prompts, e.g. {prompts[:3]}")

print("Loading Merlin model...")
model = Merlin()
model.eval()
model.cuda()

with torch.no_grad():
    text_features = model.model.encode_text(prompts)
    text_features = F.normalize(text_features, dim=-1)
print("Text features:", text_features.shape)

patient_ids = sorted(
    [f[:-len(".nii.gz")] for f in os.listdir(NIFTI_DIR) if f.endswith(".nii.gz")],
    key=lambda x: int(x),
)
datalist = [{"image": os.path.join(NIFTI_DIR, f"{pid}.nii.gz")} for pid in patient_ids]
print(f"{len(datalist)} patients to process")

dataloader = DataLoader(
    datalist=datalist,
    cache_dir=CACHE_DIR,
    batchsize=1,
    shuffle=False,
    num_workers=4,
)

results = []
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        image = batch["image"].cuda()
        image_features, ehr_features = model.model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        scores = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        pid = patient_ids[i]
        row_out = {"patient_id": pid}
        for k, s in zip(keys, scores):
            row_out[k] = float(s)
        results.append(row_out)
        print(f"[{i + 1}/{len(datalist)}] patient {pid}: done")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id"] + keys)
    writer.writeheader()
    writer.writerows(results)
print(f"Saved results to {OUT_CSV}")
