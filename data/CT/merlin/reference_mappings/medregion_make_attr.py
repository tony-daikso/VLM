import numpy as np
import torch
import json
import monai
import os
from scipy.spatial import ConvexHull, distance
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from glob import glob
import argparse
from os.path import join, basename, dirname
import pandas as pd

# pip install pandas scikit-learn matplotlib monai nibabel scikit-image

# =========================================
# 설정 부분
SPACING = (1.0, 1.0, 1.0)
SPACING_FACTOR = np.prod(SPACING)   # 1.0 * 1.0 * 1.0 for easy calculation

TOTALSEG_LOCATION_LOBE = [
    'lung_upper_lobe_left',
    'lung_lower_lobe_left',
    'lung_upper_lobe_right',
    'lung_middle_lobe_right',
    'lung_lower_lobe_right',
]
SAT_LOCATION_LOBE = [
    'left lung upper lobe',
    'left lung lower lobe',
    'right lung upper lobe',
    'right lung middle lobe',
    'right lung lower lobe',
]

LOCATION_LR = ["left_lung", "right_lung"] # 공통

# monai transform 객체는 미리 생성해서 재사용
monai_loader = monai.transforms.Compose([
    monai.transforms.LoadImaged(keys=["mask"], ensure_channel_first=True),
    monai.transforms.Rotate90d(keys=["mask"], k=3, spatial_axes=(0, 1)),
    monai.transforms.Spacingd(keys=["mask"], pixdim=SPACING, mode=("nearest")),
])
keep_largest = monai.transforms.KeepLargestConnectedComponent(applied_labels=1)

# =========================================
# 함수 정의

def load_mask(mask_path):
    try:
        dictionary = monai_loader({"mask": mask_path})
        mask = dictionary['mask'].long()
        return mask.squeeze()
    except Exception as e:
        print(f"Cannot load {mask_path}: {e}")
        return torch.zeros(1, 1, 1, dtype=torch.long)  # 빈 mask (모양 유의!)

def measure_volume(mask):
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    return float(round(np.sum(mask_np) * SPACING_FACTOR / 1000, 3))  # ml

def measure_max_diameter(mask):
    # mask: 3D numpy array or torch.Tensor (binary)
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    coords = np.argwhere(mask_np > 0)
    if coords.shape[0] < 2:
        return 0.0

    # bounding box: min/max for each axis
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    lengths = (max_coords - min_coords)
    
    # bounding box 기준 최대 치수
    max_diameter = lengths.max() * SPACING_FACTOR

    return float(round(max_diameter, 3))


def totalseg_process_patient(p_path):
    p_path.split('/')[-1]
    result = {
        p_path.split('/')[-1]: {
            "lung_nodules": {},
            "pleural_effusion": {},
            "pericardial_effusion": {},
            "kidney_cyst": {},
            "organ_volumes": {
                "lung_upper_lobe_right": 0,
                "lung_middle_lobe_right": 0,
                "lung_lower_lobe_right": 0,
                "lung_upper_lobe_left": 0,
                "lung_lower_lobe_left": 0,
                "heart_atrium_left": 0,
                "heart_atrium_right": 0,
                "heart_ventricle_left": 0,
                "heart_ventricle_right": 0,
                "liver": 0,
                "kidney_left": 0,
                "kidney_right": 0
            }
        }
    }

    try:
        # Lung nodules        
        nodule_mask = load_mask(os.path.join(p_path, 'lung_nodules.nii.gz'))
        for lobe_fname in TOTALSEG_LOCATION_LOBE:
            lobe_mask = load_mask(os.path.join(p_path, lobe_fname + ".nii.gz"))
            # print(f"lobe mask shape for {lobe_fname}:", lobe_mask.shape)
            nodule_in_lobe = nodule_mask * lobe_mask
            single_nodule = keep_largest(nodule_in_lobe)
            vol = measure_volume(single_nodule)
            dia = measure_max_diameter(single_nodule)
            result[p_path.split('/')[-1]]["lung_nodules"][lobe_fname] = {"volume (ml)": vol, "diameters (mm)": dia}
    except Exception as e:
        print(f"Error Lung {p_path}: {e}")

    try:
        # Pleural effusion
        pleural_effusion_mask = load_mask(os.path.join(p_path, 'pleural_effusion.nii.gz'))
        for loc_fname in LOCATION_LR:
            loc_mask = load_mask(os.path.join(p_path, loc_fname + ".nii.gz"))
            lesion_in_loc = pleural_effusion_mask * loc_mask
            single_lesion = keep_largest(lesion_in_loc)
            vol = measure_volume(single_lesion)
            result[p_path.split('/')[-1]]["pleural_effusion"][loc_fname] = {"volume (ml)": vol}

        # Pericardial effusion
        pericardial_effusion_mask = load_mask(os.path.join(p_path, 'pericardial_effusion.nii.gz'))
        single_lesion = keep_largest(pericardial_effusion_mask)
        vol = measure_volume(single_lesion)
        result[p_path.split('/')[-1]]["pericardial_effusion"] = {"volume (ml)": vol}
    except Exception as e:
        print(f"Error Effusion {p_path}: {e}")

    try:
        # Kidney cyst
        for side in ["left", "right"]:
            cyst_mask = load_mask(os.path.join(p_path, f'kidney_cyst_{side}.nii.gz'))
            single_lesion = keep_largest(cyst_mask)
            vol = measure_volume(single_lesion)
            dia = measure_max_diameter(single_lesion)
            result[p_path.split('/')[-1]]["kidney_cyst"][f"{side}_kidney"] = {"volume (ml)": vol, "diameters (mm)": dia}
    except Exception as e:
        print(f"Error Kidney Cyst {p_path}: {e}")

    try:
        # setting TotalSegmentator organ names
        organ_names = [
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "lung_upper_lobe_left", "lung_lower_lobe_left",
            "heart_atrium_left", "heart_ventricle_left", "heart_atrium_right", "heart_ventricle_right", # totalseg는 독릳적으로 segmask를 뽑아놔야함.
            "liver",
            "kidney_left", "kidney_right"
        ]
        for organ in organ_names:
            organ_mask = load_mask(os.path.join(p_path, f"{organ}.nii.gz"))
            vol = measure_volume(organ_mask)
            result[p_path.split('/')[-1]]["organ_volumes"][organ] = vol
    except Exception as e:
        print(f"Error processing {p_path}: {e}")

    return result

def sat_process_patient(p_path):
    p_path = p_path.replace('.nii.gz', '')

    result = {
        p_path.split('/')[-1]: {
            "lung nodule": {},
            "lung effusion": {},
            "kidney cyst": {},
            "organ_volumes": {
                "right lung upper lobe": 0,
                "right lung middle lobe": 0,
                "right lung lower lobe": 0,
                "left lung upper lobe": 0,
                "left lung lower lobe": 0,                
                "left heart atrium": 0,
                "left heart ventricle": 0,
                "right heart atrium": 0,
                "right heart ventricle": 0,
                "liver": 0,
                "left kidney": 0,
                "right kidney": 0
            }
        }
    }
    try:
        # Lung nodule
        nodule_mask = load_mask(os.path.join(p_path, 'lung nodule.nii.gz'))
        for lobe_fname in SAT_LOCATION_LOBE:
            lobe_mask = load_mask(os.path.join(p_path, lobe_fname + ".nii.gz"))
            # print(f"lobe mask shape for {lobe_fname}:", lobe_mask.shape)
            nodule_in_lobe = nodule_mask * lobe_mask
            single_nodule = keep_largest(nodule_in_lobe)
            vol = measure_volume(single_nodule)
            dia = measure_max_diameter(single_nodule)
            result[p_path.split('/')[-1]]["lung nodule"][lobe_fname] = {"volume (ml)": vol, "diameters (mm)": dia}
    except Exception as e:
        print(f"Error Lung {p_path}: {e}")

    try:
        # Lung effusion
        lung_effusion_mask = load_mask(os.path.join(p_path, 'lung effusion.nii.gz'))
        single_lesion = keep_largest(lung_effusion_mask)
        vol = measure_volume(single_lesion)
        result[p_path.split('/')[-1]]["lung effusion"] = {"volume (ml)": vol}
    except Exception as e:
        print(f"Error Effusion {p_path}: {e}")

    try:
        # Kidney cyst
        for side in ["left", "right"]:
            cyst_mask = load_mask(os.path.join(p_path, f'{side} kidney cyst.nii.gz'))
            single_lesion = keep_largest(cyst_mask)
            vol = measure_volume(single_lesion)
            dia = measure_max_diameter(single_lesion)
            result[p_path.split('/')[-1]]["kidney cyst"][f"{side} kidney"] = {"volume (ml)": vol, "diameters (mm)": dia}
    except Exception as e:
        print(f"Error Kidney Cyst {p_path}: {e}")

    try:
        # setting SAT organ names
        organ_names = [
            "right lung upper lobe", "right lung middle lobe", "right lung lower lobe", "left lung upper lobe", "left lung lower lobe",
            "left heart atrium", "left heart ventricle", "right heart atrium", "right heart ventricle",
            "left lobe of liver", "right lobe of liver",
            "left kidney", "right kidney"
        ]
        for organ in organ_names:
            organ_mask = load_mask(os.path.join(p_path, f"{organ}.nii.gz"))
            vol = measure_volume(organ_mask)
            result[p_path.split('/')[-1]]["organ_volumes"][organ] = vol
        
        # merge liver volume
        result[p_path.split('/')[-1]]["organ_volumes"]["liver"] = result[p_path.split('/')[-1]]["organ_volumes"].pop("left lobe of liver") + result[p_path.split('/')[-1]]["organ_volumes"].pop("right lobe of liver")
        
    except Exception as e:
        print(f"Error processing {p_path}: {e}")

    return result


# =========================================
# 메인

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--target_csv', type=str, required=True)
    parser.add_argument('--seg_type', type=str, default="totalseg", choices=["totalseg", "sat"])
    parser.add_argument('--seg_base_path', type=str, default="/workspace/5.Lung/MedRegion-CT-Public/dataset/sample_dataset/Totalsegmentor_masks", required=True)
    args = parser.parse_args()
    

    print("Target CSV:", args.target_csv)
    df = pd.read_csv(args.target_csv)
    print("TOTAL Number of input images:", len(df))

    path_list = df['id'].map(lambda x: os.path.join(args.seg_base_path, x.replace(".nii.gz", ""))).tolist()
    print("path_list[0] == ", path_list[0])
    
    if args.end == -1:
        path_list = path_list[args.start:]
    else:
        path_list = path_list[args.start:args.end]

    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    print("CPU ===>", cpu_count())
    use_cpu = cpu_count()//2
    all_results = {}

    with Pool(processes=use_cpu) as pool:
        results = []
        if args.seg_type == "totalseg":
            for res in tqdm(pool.imap_unordered(totalseg_process_patient, path_list), total=len(path_list), desc="Processing"):
                results.append(res)
        
        elif args.seg_type == "sat":
            for res in tqdm(pool.imap_unordered(sat_process_patient, path_list), total=len(path_list), desc="Processing"):
                results.append(res)

        else:
            raise NotImplementedError("Only 'totalseg' is implemented in this script.")

    for r in results:
        all_results.update(r)

    with open(args.save, "w") as f:
        json.dump(all_results, f, indent=4)

    print("Finished processing all patients.")
