import copy
import functools
import hashlib
import json
import os
import pickle
import time
import traceback
import warnings
from collections import Counter

import numpy as np
import torch
import torchio as tio
from torch.utils import data
from tqdm import tqdm

from pillar.utils.logging import logger
from pillar.datasets import image_loaders as loaders
from pillar.datasets.nlst_utils import (
    LOAD_FAIL_MSG,
    METAFILE_NOTFOUND_ERR,
    VOXEL_SPACING,
    get_censoring_dist,
)
import rve
import torch.nn.functional as F
import pandas as pd

is_master = None


def get_is_master():
    return is_master


# Timing debug flag
ENABLE_TIMING = os.environ.get("ENABLE_DATASET_TIMING", "0") == "1"


def timing_print(msg):
    if ENABLE_TIMING:
        print(msg)


DIR = "data/nlst_dataset/"

METADATA_FILENAME = {"google_test": f"{DIR}full_nlst_google.json"}

GOOGLE_SPLITS_FILENAME = f"{DIR}/shetty_google_data_splits.p"

CORRUPTED_PATHS = f"{DIR}/corrupted_img_paths.pkl"

CT_ITEM_KEYS = [
    "pid",
    "exam",
    "series",
    "y_seq",
    "y_mask",
    "time_at_event",
    "cancer_laterality",
    "has_annotation",
    "origin_dataset",
]

RACE_ID_KEYS = {
    1: "white",
    2: "black",
    3: "asian",
    4: "american_indian_alaskan",
    5: "native_hawaiian_pacific",
    6: "hispanic",
}
ETHNICITY_KEYS = {1: "Hispanic or Latino", 2: "Neither Hispanic nor Latino"}
GENDER_KEYS = {1: "Male", 2: "Female"}
EDUCAT_LEVEL = {
    1: 1,  # 8th grade = less than HS
    2: 1,  # 9-11th = less than HS
    3: 2,  # HS Grade
    4: 3,  # Post-HS
    5: 4,  # Some College
    6: 5,  # Bachelors = College Grad
    7: 6,  # Graduate School = Postrad/Prof
}

CENSORING_DIST = {
    "0": 0.9851928130104401,
    "1": 0.9748317321074379,
    "2": 0.9659923988537479,
    "3": 0.9587252204657843,
    "4": 0.9523590830936284,
    "5": 0.9461840310101468,
}

METADATA_CACHE_ROOT = "cache/nlst/metadata"


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


def cache_to_file(key, func, *args, **kwargs):
    # We turn key to str (otherwise dict is unhashablep)
    key_hash = md5(str((key, args, kwargs)))

    cache_path = f"{METADATA_CACHE_ROOT}/{key_hash}.pkl"

    # Technically without barrier it's possible for other processes to read partial cache, but the chance is small.
    if os.path.exists(cache_path):
        logger.debug(f"Metadata cache exists ({cache_path}), skipping metadata processing.")

        with open(cache_path, "rb") as f:
            data, _ = pickle.load(f)

        return data
    else:
        data = func(*args, **kwargs)

        logger.debug(f"Metadata cache does not exist, saving to {cache_path}.")

        # Only write to cache in master
        if get_is_master():
            os.makedirs(METADATA_CACHE_ROOT, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((data, (key, args, kwargs)), f)

        return data


class NLSTDataset(data.Dataset):
    def __init__(
        self,
        args,
        augmentations,
        split_group,
        input_loader="DicomGroupLoader",
        img_dir="datasets/nlst/nlst-ct-png/",
        cache_path="./datasets/nlst/cache/res128_128_200",
        img_file_type="dicom",
        use_risk_factors=False,
        tokenizer=None,
        cross_val_seed=0,
        split_type="random",
        assign_splits=False,
        split_probs=[0.6, 0.2, 0.2],
        max_followup=6,
        dataset_file_path=None,
        resample_pixel_spacing=False,
        resample_pixel_spacing_prob=0.0,
        num_images=200,
        fit_to_length=False,
        min_num_images=0,
        slice_thickness_filter=2.5,
        use_only_thin_cuts_for_ct=False,
        use_annotations=False,
        region_annotations_filepath=None,
        fix_seed_for_multi_image_augmentations=False,
        ignore_exams=[],
        compute_censoring_distribution=False,
    ):
        """
        Builds NLST Dataset from https://cdas.cancer.gov/nlst/

        Constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching

        Args:
            args (Namespace): config
            augmentations (augmentations): A augmentations object, takes in a PIL image, performs some transforms and returns a Tensor
            split_group (str): ['train'|'dev'|'test'].
            input_loader (str, optional). Defaults to "DicomGroupLoader".
            img_dir (str, optional). Defaults to "datasets/nlst/nlst-ct-png/".
            cache_path (str, optional). Defaults to "./datasets/nlst/cache/res128_128_200".
            img_file_type (str). Defaults to "dicom".
            use_risk_factors (bool, optional): Whether to feed risk factors into last FC of model. Defaults to False.
            cross_val_seed (int, optional): Seed used to generate the partition. Defaults to 0.
            split_type (str, optional): How to split dataset if assign_split = True. Usage: ['random', 'institution_split']. Defaults to "random".
            assign_splits (bool, optional): Whether to assign different splits than those predetermined in dataset. Defaults to False.
            split_probs (list, optional): Split probs for datasets without fixed train dev test. Defaults to [0.6, 0.2, 0.2].
            max_followup (int, optional): Max followup to predict over. Defaults to 6.
            dataset_file_path (str, optional): Path to dataset file either as json or csv. Defaults to None.
            resample_pixel_spacing (bool, optional): Whether to resample pixel spacing into fixed dimensions. Defaults to False.
            resample_pixel_spacing_prob (_type_, optional): Probability of resampling pixel spacing into fixed dimensions. 1 when eval and using resampling. Defaults to 0..
            num_images (int, optional): In multi image setting, the number of images per single sample. Defaults to 200.
            fit_to_length (bool, optional): Whether to fit num slices using padding and slice sampling. Defaults to False.
            min_num_images (int, optional): In multi image setting, the min number of images per single sample. Defaults to 0.
            slice_thickness_filter (float, optional): Slice thickness using, if restricting to specific thickness value. Defaults to 2.5.
            use_only_thin_cuts_for_ct (bool, optional): Wether to use image series with thinnest cuts only. Defaults to False.
            use_annotations (bool, optional): Whether to use image annotations (pixel labels) in modeling. Defaults to False.
            region_annotations_filepath (str, optional): Path to annotations file. Defaults to None.
            fix_seed_for_multi_image_augmentations (bool, optional): Whether to use the same seed (same random augmentations) for multi image inputs. Defaults to False.
            compute_censoring_distribution (bool, optional): Whether to re-compute the censoring distribution for the training set. The censoring distribution will also be prined out. Typically the distribution is computed once and saved into the code. Defaults to False.
        """
        super(NLSTDataset, self).__init__()

        self.split_group = split_group
        self.args = args
        self.num_images = num_images  # number of slices in each volume
        self.img_dir = img_dir
        self.max_followup = max_followup
        self.img_file_type = img_file_type
        self.use_risk_factors = use_risk_factors
        self.cross_val_seed = cross_val_seed
        self.split_type = split_type
        self.assign_splits = assign_splits
        self.split_probs = split_probs
        self.use_only_thin_cuts_for_ct = use_only_thin_cuts_for_ct
        self.slice_thickness_filter = slice_thickness_filter
        self.min_num_images = min_num_images
        self.dataset_file_path = dataset_file_path
        self.resample_pixel_spacing = resample_pixel_spacing
        self.resample_pixel_spacing_prob = resample_pixel_spacing_prob
        self.fit_to_length = fit_to_length
        self.use_annotations = use_annotations
        self.fix_seed_for_multi_image_augmentations = fix_seed_for_multi_image_augmentations
        self.ignore_exams = ignore_exams

        self.input_loader = loaders.ResampleDicomGroupLoader(
            cache_path, augmentations, group_key_name="exam_str", args=args
        )

        self.always_resample_pixel_spacing = split_group in ["dev", "test"]

        self.resample_transform = tio.transforms.Resample(target=VOXEL_SPACING)
        # self.img_size = self.args.dataset.img_size
        self.img_size = [256, 256]
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=tuple(self.img_size + [num_images]), padding_mode=0
        )

        logger.debug(f"Image size: {self.img_size}")

        self.tokenizer = tokenizer

        if use_annotations:
            assert region_annotations_filepath, "ANNOTATIONS METADATA FILE NOT SPECIFIED"
            self.annotations_metadata = json.load(open(region_annotations_filepath, "r"))

        # Without cache:
        # self.dataset = self.create_dataset(split_group)
        # With cache:
        self.dataset = cache_to_file(
            # {"name": "create_dataset", "config": self.args.dataset},
            {"name": "create_dataset", "config": None},
            self.create_dataset,
            split_group,
        )
        if len(self.dataset) == 0:
            return

        # with open("datasets/nlst/image_to_captions.pickle", "rb") as f:
        #     image_to_captions = pickle.load(f)

        # self.image_to_captions = {}
        # for k, v in image_to_captions.items():
        #     series = str(k.split('/')[-1])
        #     self.image_to_captions[series] = v

        logger.debug(self.get_summary_statement(self.dataset, split_group))

        # create label distribution
        # Note : performs a pass of the dataset to create label set
        dist_key = "y"
        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1.0 / len(label_counts)
        label_weights = {label: weight_per_label / count for label, count in label_counts.items()}

        # report dataset statistics
        logger.debug("Class counts are: {}".format(label_counts))
        logger.debug("Label weights are {}".format(label_weights))
        self.weights = [label_weights[d[dist_key]] for d in self.dataset]

        info = {}

        self.no_captions = set()

        if split_group == "train":
            # Note that even for evaluation on dev, we want to use the censoring distribution on training set.
            # On test it depends. For now, we are also using the censoring distribution on training set.

            if compute_censoring_distribution:
                logger.debug("Computing censoring distribution...")
                censoring_distribution = get_censoring_dist(self.dataset)
                logger.debug(f"Censoring distribution: {censoring_distribution}")
            else:
                censoring_distribution = CENSORING_DIST

            info["censoring_distribution"] = censoring_distribution

        self.info = info

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in metadata.json.

        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
        # Dataset reads a metadata file, that contains path to all images
        try:
            metadata_json = json.load(open(self.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(self.dataset_file_path, e))

        self.corrupted_paths = self.CORRUPTED_PATHS["paths"]
        self.corrupted_series = self.CORRUPTED_PATHS["series"]

        # define manual splits
        if self.assign_splits:
            np.random.seed(self.cross_val_seed)
            self.assign_dataset_splits(metadata_json)

        dataset = []

        for mrn_row in tqdm(metadata_json, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if not split == split_group:
                continue

            for exam_dict in exams:
                if self.use_only_thin_cuts_for_ct and split_group in [
                    "train",
                    "dev",
                ]:
                    thinnest_series_id = self.get_thinnest_cut(exam_dict)

                elif split == "test" and self.assign_splits:
                    thinnest_series_id = self.get_thinnest_cut(exam_dict)

                elif split == "test":
                    google_series = list(self.GOOGLE_SPLITS[pid]["exams"])
                    nlst_series = list(exam_dict["image_series"].keys())
                    thinnest_series_id = [s for s in nlst_series if s in google_series]
                    assert len(thinnest_series_id) < 2
                    if len(thinnest_series_id) > 0:
                        thinnest_series_id = thinnest_series_id[0]
                    elif len(thinnest_series_id) == 0:
                        if self.assign_splits:
                            thinnest_series_id = self.get_thinnest_cut(exam_dict)
                        else:
                            continue

                for series_id, series_dict in exam_dict["image_series"].items():
                    if self.skip_sample(series_dict, pt_metadata):
                        continue

                    if self.use_only_thin_cuts_for_ct and (not series_id == thinnest_series_id):
                        continue

                    # create volume and labels
                    sample = self.get_volume_dict(series_id, series_dict, exam_dict, pt_metadata, pid, split)
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        return dataset

    def get_thinnest_cut(self, exam_dict):
        # volume that is not thin cut might be the one annotated; or there are multiple volumes with same num slices, so:
        # use annotated if available, otherwise use thinnest cut
        possibly_annotated_series = [s in self.annotations_metadata for s in list(exam_dict["image_series"].keys())]
        series_lengths = [
            len(exam_dict["image_series"][series_id]["paths"]) for series_id in exam_dict["image_series"].keys()
        ]
        thinnest_series_len = max(series_lengths)
        thinnest_series_id = [k for k, v in exam_dict["image_series"].items() if len(v["paths"]) == thinnest_series_len]
        if any(possibly_annotated_series):
            thinnest_series_id = list(exam_dict["image_series"].keys())[possibly_annotated_series.index(1)]
        else:
            thinnest_series_id = thinnest_series_id[0]
        return thinnest_series_id

    def skip_sample(self, series_dict, pt_metadata):
        series_data = series_dict["series_data"]
        # check if screen is localizer screen or not enough images
        is_localizer = self.is_localizer(series_data)

        # check if restricting to specific slice thicknesses
        slice_thickness = series_data["reconthickness"][0]
        wrong_thickness = (self.slice_thickness_filter is not None) and (
            slice_thickness > self.slice_thickness_filter or (slice_thickness < 0)
        )

        # check if valid label (info is not missing)
        screen_timepoint = series_data["study_yr"][0]
        bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # invalid label
        if not bad_label:
            y, _, _, time_at_event = self.get_label(pt_metadata, screen_timepoint)
            invalid_label = (y == -1) or (time_at_event < 0)
        else:
            invalid_label = False

        insufficient_slices = len(series_dict["paths"]) < self.min_num_images

        if is_localizer or wrong_thickness or bad_label or invalid_label or insufficient_slices:
            return True
        else:
            return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, pt_metadata, pid, split):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        device = series_data["manufacturer"][0]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if series_id in self.corrupted_series:
            if any([path in self.corrupted_paths for path in img_paths]):
                uncorrupted_imgs = np.where([path not in self.corrupted_paths for path in img_paths])[0]
                img_paths = np.array(img_paths)[uncorrupted_imgs].tolist()
                slice_locations = np.array(slice_locations)[uncorrupted_imgs].tolist()

        sorted_img_paths, sorted_slice_locs = self.order_slices(img_paths, slice_locations)

        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, screen_timepoint)

        exam_int = int("{}{}{}".format(int(pid), int(screen_timepoint), int(series_id.split(".")[-1][-3:])))
        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": exam_int,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "study": series_data["studyuid"][0],
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "device": device,
            "institution": pt_metadata["cen"][0],
            "cancer_laterality": self.get_cancer_side(pt_metadata),
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"] + [series_dict["slice_thickness"]],
            "slice_thickness": self.get_slice_thickness_class(series_dict["slice_thickness"]),
        }

        if self.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(pt_metadata, screen_timepoint, return_dict=False)

        return sample

    def check_label(self, pt_metadata, screen_timepoint):
        valid_days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.max_followup
        y_seq = np.zeros(self.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.max_followup - 1)
        y_mask = np.array([1] * (time_at_event + 1) + [0] * (self.max_followup - (time_at_event + 1)))
        assert len(y_mask) == self.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def is_localizer(self, series_dict):
        is_localizer = (
            (series_dict["imageclass"][0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer

    def get_cancer_side(self, pt_metadata):
        """
        Return if cancer in left or right

        right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
        left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
        else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
        """
        right_keys = ["locrhil", "locrlow", "locrmid", "locrmsb", "locrup"]
        left_keys = ["loclup", "loclmsb", "locllow", "loclhil", "loclin"]
        other_keys = ["loccar", "locmed", "locoth", "locunk"]

        right = any([pt_metadata[key][0] > 0 for key in right_keys])
        left = any([pt_metadata[key][0] > 0 for key in left_keys])
        other = any([pt_metadata[key][0] > 0 for key in other_keys])

        return np.array([int(right), int(left), int(other)])

    def order_slices(self, img_paths, slice_locations):
        sorted_ids = np.argsort(slice_locations)
        sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
        sorted_slice_locs = np.sort(slice_locations).tolist()

        if not sorted_img_paths[0].startswith(self.img_dir):
            sorted_img_paths = [
                self.img_dir + path[path.find("nlst-ct-png") + len("nlst-ct-png") :] for path in sorted_img_paths
            ]
        if (
            self.img_file_type == "dicom"
        ):  # ! NOTE: removing file extension affects get_ct_annotations mapping path to annotation
            sorted_img_paths = [path.replace("nlst-ct-png", "nlst-ct").replace(".png", "") for path in sorted_img_paths]

        return sorted_img_paths, sorted_slice_locs

    def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
        age_at_randomization = pt_metadata["age"][0]
        days_since_randomization = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        current_age = age_at_randomization + days_since_randomization // 365

        age_start_smoking = pt_metadata["smokeage"][0]
        age_quit_smoking = pt_metadata["age_quit"][0]
        years_smoking = pt_metadata["smokeyr"][0]
        is_smoker = pt_metadata["cigsmok"][0]

        years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

        education = pt_metadata["educat"][0] if pt_metadata["educat"][0] != -1 else pt_metadata["educat"][0]

        race = pt_metadata["race"][0] if pt_metadata["race"][0] != -1 else 0
        race = 6 if pt_metadata["ethnic"][0] == 1 else race
        ethnicity = pt_metadata["ethnic"][0]

        weight = pt_metadata["weight"][0] if pt_metadata["weight"][0] != -1 else 0
        height = pt_metadata["height"][0] if pt_metadata["height"][0] != -1 else 0
        bmi = weight / (height**2) * 703 if height > 0 else 0  # inches, lbs

        prior_cancer_keys = [
            "cancblad",
            "cancbrea",
            "canccerv",
            "canccolo",
            "cancesop",
            "canckidn",
            "canclary",
            "canclung",
            "cancoral",
            "cancnasa",
            "cancpanc",
            "cancphar",
            "cancstom",
            "cancthyr",
            "canctran",
        ]
        cancer_hx = any([pt_metadata[key][0] == 1 for key in prior_cancer_keys])
        family_hx = any([pt_metadata[key][0] == 1 for key in pt_metadata if key.startswith("fam")])

        risk_factors = {
            "age": current_age,
            "race": race,
            "race_name": RACE_ID_KEYS.get(pt_metadata["race"][0], "UNK"),
            "ethnicity": ethnicity,
            "ethnicity_name": ETHNICITY_KEYS.get(ethnicity, "UNK"),
            "education": education,
            "bmi": bmi,
            "cancer_hx": cancer_hx,
            "family_lc_hx": family_hx,
            "copd": pt_metadata["diagcopd"][0],
            "is_smoker": is_smoker,
            "smoking_intensity": pt_metadata["smokeday"][0],
            "smoking_duration": pt_metadata["smokeyr"][0],
            "years_since_quit_smoking": years_since_quit_smoking,
            "weight": weight,
            "height": height,
            "gender": GENDER_KEYS.get(pt_metadata["gender"][0], "UNK"),
        }

        if return_dict:
            return risk_factors
        else:
            return np.array([v for v in risk_factors.values() if not isinstance(v, str)])

    def assign_dataset_splits(self, meta):
        if self.split_type == "institution_split":
            self.assign_institutions_splits(meta)
        elif self.split_type == "random":
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(["train", "dev", "test"], p=self.split_probs)

    def assign_institutions_splits(self, meta):
        institutions = set([m["pt_metadata"]["cen"][0] for m in meta])
        institutions = sorted(institutions)
        institute_to_split = {
            cen: np.random.choice(["train", "dev", "test"], p=self.split_probs) for cen in institutions
        }
        for idx in range(len(meta)):
            meta[idx]["split"] = institute_to_split[meta[idx]["pt_metadata"]["cen"][0]]

    @property
    @functools.cache
    def METADATA_FILENAME(self):
        return METADATA_FILENAME["google_test"]

    @property
    @functools.cache
    def CORRUPTED_PATHS(self):
        return pickle.load(open(CORRUPTED_PATHS, "rb"))

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(split_group, len(dataset), len(exams), len(patients), class_balance)
        statement += "\n" + "Censor Times: {}".format(Counter([d["time_at_event"] for d in dataset]))
        statement
        return statement

    @property
    @functools.cache
    def GOOGLE_SPLITS(self):
        return pickle.load(open(GOOGLE_SPLITS_FILENAME, "rb"))

    def _remove_empty_annotations(self, series):
        if series in self.annotations_metadata:
            self.annotations_metadata[series] = {
                k: v for k, v in self.annotations_metadata[series].items() if len(v) > 0
            }

    def _extract_annotation_paths(self, series, base_ann_paths):
        if series in self.annotations_metadata:
            if self.img_file_type == "dicom":

                def extract_path_fn(path):
                    return os.path.basename(path)
            else:

                def extract_path_fn(path):
                    return os.path.splitext(os.path.basename(path))[0]

            annotation_paths = [
                {"image_annotations": self.annotations_metadata[series].get(extract_path_fn(path), None)}
                for path in base_ann_paths
            ]
        else:
            annotation_paths = [{"image_annotations": None} for path in base_ann_paths]

        return annotation_paths

    def get_ct_annotations(self, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        Args:
            sample: A dictionary with img paths, label,
                and additional information regarding exam or participant
        Returns:
            The sample with annotations updated
        """
        # correct empty lists of annotations
        self._remove_empty_annotations(sample["series"])

        # extract annotation paths
        # e.g. remove extensions, remove paths that don't have annotations
        sample["annotations"] = self._extract_annotation_paths(sample["series"], sample["paths"])

        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        t_start = time.time() if ENABLE_TIMING else None
        try:
            sample = self.dataset[index]
        except IndexError:
            warnings.warn(f"IndexError: {index} out of {len(self.dataset)}")
            # traceback.print_stack()
            return None
        if ENABLE_TIMING:
            t_sample_fetch = time.time()
            timing_print(f"[TIMING] Sample fetch: {t_sample_fetch - t_start:.4f}s")

        if sample["exam"] in self.ignore_exams:
            sample = self.dataset[index + 1]
            # Ignore this exam by returning None
            # images = torch.zeros((1, 96, 256, 256))
            # caption = "INVALID SAMPLE."
            # texts = self.tokenizer([caption])[0]
            # item = {}
            # item['x'] = images
            # item['images'] = images
            # item['texts'] = texts
            # item['y'] = [0, 0, 0, 0, 0, 0]
            # item['mask'] = torch.zeros((1, 96, 256, 256))
            # item['has_annotation'] = False
            # item['volume_annotations'] =
            # return item
            # return images, texts

        # get path to ct annotations
        t_before_annotations = time.time() if ENABLE_TIMING else None
        if self.use_annotations:
            sample = self.get_ct_annotations(sample)
        if ENABLE_TIMING:
            t_after_annotations = time.time()
            timing_print(f"[TIMING] get_ct_annotations: {t_after_annotations - t_before_annotations:.4f}s")

        # get images from paths for multi image input
        try:
            item = {}

            t_before_load = time.time() if ENABLE_TIMING else None
            input_dict = self.get_images(sample["paths"], sample)
            if ENABLE_TIMING:
                t_after_load = time.time()
                timing_print(f"[TIMING] get_images: {t_after_load - t_before_load:.4f}s")

            t_before_processing = time.time() if ENABLE_TIMING else None
            x = input_dict["input"]
            item["x"] = x
            item["mask"] = input_dict["mask"]

            if self.use_annotations and input_dict["mask"] is not None:
                mask = torch.abs(input_dict["mask"])
                mask_area = mask.sum(dim=(-1, -2))
                item["volume_annotations"] = mask_area[0] / max(1, mask_area.sum())
                item["annotation_areas"] = mask_area[0] / (mask.shape[-2] * mask.shape[-1])
                mask_area = mask_area.unsqueeze(-1).unsqueeze(-1)
                mask_area[mask_area == 0] = 1
                item["image_annotations"] = mask / mask_area
                item["has_annotation"] = item["volume_annotations"].sum() > 0

            if self.use_risk_factors:
                item["risk_factors"] = sample["risk_factors"]

            item["x"] = x
            item["y"] = sample["y"]
            item["index"] = index
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            if ENABLE_TIMING:
                t_end = time.time()
                timing_print(f"[TIMING] Post-processing: {t_end - t_before_processing:.4f}s")
                timing_print(f"[TIMING] Total __getitem__: {t_end - t_start:.4f}s")

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def read_volume_from_disk(self, paths, sample, use_annotations=False):
        return self.input_loader.get_volume(paths, sample, use_annotations=use_annotations)

    def read_input_from_disk(self, paths, sample):
        # get images for multi image input
        s = copy.deepcopy(sample)
        input_dicts = []
        for path_idx, path in enumerate(paths):
            if self.use_annotations:
                s["annotations"] = sample["annotations"][path_idx]

            first_image_in_group = path_idx == 0
            input_dicts.append(self.input_loader.get_image(path, s, first_image_in_group=first_image_in_group))
        return input_dicts

    def get_images(self, paths, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        if self.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        input_dicts = self.read_input_from_disk(paths, sample)

        # Need to close the readers/writers since we finish reading from a group
        if "cache" in self.input_loader.__dict__:
            self.input_loader.cache.close_all_groups()

        out_dict = self.resample_reshape_pad_input(input_dicts, sample)
        return out_dict

    def resample_reshape_pad_input(self, input_dicts, sample, write_to_disk=False):
        out_dict = {}
        # images = [torch.tensor(i["input"], dtype=torch.float32).unsqueeze(0) for i in input_dicts]
        images = [i["input"] for i in input_dicts]
        input_arr = self.reshape_images(images)

        if self.use_annotations:
            masks = [i["mask"] for i in input_dicts]
            # masks = [torch.tensor(i["mask"], dtype=torch.float32).unsqueeze(0) for i in input_dicts]
            mask_arr = self.reshape_images(masks)
        else:
            mask_arr = None

        # resample pixel spacing
        resample_now = self.resample_pixel_spacing_prob > np.random.uniform()
        if self.always_resample_pixel_spacing or resample_now:
            spacing = torch.tensor(sample["pixel_spacing"] + [1])
            input_arr = tio.ScalarImage(
                affine=torch.diag(spacing),
                # cthw -> chwt
                tensor=input_arr.permute(0, 2, 3, 1),
            )
            input_arr = self.resample_transform(input_arr)
            input_arr = self.padding_transform(input_arr.data)

            if self.use_annotations:
                mask_arr = tio.ScalarImage(
                    affine=torch.diag(spacing),
                    # cthw -> chwt
                    tensor=mask_arr.permute(0, 2, 3, 1),
                )
                mask_arr = self.resample_transform(mask_arr)
                mask_arr = self.padding_transform(mask_arr.data)

        out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)
        if self.use_annotations:
            out_dict["mask"] = mask_arr.data.permute(0, 3, 1, 2)
        return out_dict

    def reshape_images(self, images):
        # check if images are none

        images = [im.unsqueeze(0) for im in images]
        images = torch.cat(images, dim=0)
        # convert from (T, C, H, W) to (C, T, H, W)
        if len(images.shape) == 4:
            images = images.permute(1, 0, 2, 3)
        return images

    def get_slice_thickness_class(self, thickness):
        BINS = [1, 1.5, 2, 2.5]
        for i, tau in enumerate(BINS):
            if thickness <= tau:
                return i
        if self.slice_thickness_filter is not None:
            raise ValueError("THICKNESS > 2.5")
        return 4


class RVECacheNLST(NLSTDataset):
    def __init__(
        self,
        *args,
        anatomy="chest_ct",
        num_images=192,
        windows="all",
        rve_cache_dir="data/nlst_r256x256x192_1.25_ultrafast_libx265/cache_extracted/nlst_r256x256x192_1.25_ultrafast_libx265",
        rve_masks_dir="data/nlst_r256x256x192_1.25_ultrafast_libx265/masks_extracted/bounding_masks_rve_resampled/nlst_masks_resampled",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.anatomy = anatomy
        self.rve_cache_dir = rve_cache_dir
        self.rve_masks_dir = rve_masks_dir
        self.resample_input = False
        self.write_volume_to_disk = False
        self.mapping = pd.read_csv(os.path.join(self.rve_cache_dir, "mapping.csv"))
        self.mapping["series"] = self.mapping["source_path"].apply(lambda x: x.split("/")[-1])
        series_set = set(self.mapping["series"])
        dataset = []
        for row in self.dataset:
            if row["series"] in series_set:
                dataset.append(row)
        self.rve_dataset = dataset

        # create label distribution
        # Note : performs a pass of the dataset to create label set
        dist_key = "y"
        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1.0 / len(label_counts)
        label_weights = {label: weight_per_label / count for label, count in label_counts.items()}
        self.weights = [label_weights[d[dist_key]] for d in self.rve_dataset]
        logger.info(f"Loaded {len(self.rve_dataset)} samples from RVE cache")
        self.num_images = num_images

    def __len__(self):
        return len(self.rve_dataset)

    def __getitem__(self, index):
        t_start = time.time() if ENABLE_TIMING else None
        try:
            sample = self.rve_dataset[index]
        except IndexError:
            warnings.warn(f"IndexError: {index} out of {len(self.rve_dataset)}")
            return None
        if ENABLE_TIMING:
            t_sample_fetch = time.time()
            timing_print(f"[TIMING] Sample fetch: {t_sample_fetch - t_start:.4f}s")

        if sample["exam"] in self.ignore_exams:
            sample = self.rve_dataset[index + 1]

        # No fallback: directly attempt to load cached RVE volume/mask
        # Any failure in loading will raise and propagate to caller
        item = {}

        t_before_load = time.time() if ENABLE_TIMING else None
        input_dict = self.get_images(sample["paths"], sample)
        if ENABLE_TIMING:
            t_after_load = time.time()
            timing_print(f"[TIMING] get_images: {t_after_load - t_before_load:.4f}s")

        if input_dict["input"].shape != input_dict["mask"].shape:
            print(sample["paths"])
            print(f"Volume shape: {input_dict['input'].shape}")
            print(f"Mask shape: {input_dict['mask'].shape}")
            return self.__getitem__(index + 1)

        t_before_processing = time.time() if ENABLE_TIMING else None
        x = input_dict["input"]
        item["x"] = x
        item["mask"] = input_dict["mask"]
        item["has_annotation"] = False

        if self.use_annotations and input_dict["mask"] is not None:
            mask = torch.abs(input_dict["mask"])
            mask_area = mask.sum(dim=(-1, -2))
            item["volume_annotations"] = mask_area[0] / max(1, mask_area.sum())
            item["annotation_areas"] = mask_area[0] / (mask.shape[-2] * mask.shape[-1])
            mask_area = mask_area.unsqueeze(-1).unsqueeze(-1)
            mask_area[mask_area == 0] = 1
            item["image_annotations"] = mask / mask_area
            item["has_annotation"] = item["volume_annotations"].sum() > 0

        if self.use_risk_factors:
            item["risk_factors"] = sample["risk_factors"]

        item["accession"] = sample["accession"]
        item["exam"] = sample["exam"]
        item["x"] = x
        item["y"] = sample["y"]
        item["index"] = index
        item["anatomy"] = self.anatomy
        for key in CT_ITEM_KEYS:
            if key in sample:
                item[key] = sample[key]

        if ENABLE_TIMING:
            t_end = time.time()
            timing_print(f"[TIMING] Post-processing: {t_end - t_before_processing:.4f}s")
            timing_print(f"[TIMING] Total __getitem__: {t_end - t_start:.4f}s")

        return item

    def read_resample_volume_from_disk(self, paths, sample):
        """Read cached volume and mask from RVE cache directory"""
        t_start = time.time() if ENABLE_TIMING else None
        series_id = sample["series"]
        volume_path = os.path.join(self.rve_cache_dir, f"{series_id}.1.0")

        t_before_volume_load = time.time() if ENABLE_TIMING else None
        # Load the volume using rve
        if os.path.exists(volume_path):
            # print(f"Loading volume | sample {sample['accession']} {sample['series']} {sample['screen_timepoint']}")
            volume = rve.load_sample(volume_path, use_hardware_acceleration=False)
        else:
            raise FileNotFoundError(
                f"Volume not found | sample {sample['accession']} {sample['series']} {sample['screen_timepoint']}"
            )
        if ENABLE_TIMING:
            t_after_volume_load = time.time()
            timing_print(f"[TIMING]   - rve.load_sample (volume): {t_after_volume_load - t_before_volume_load:.4f}s")

        # Load mask if it exists and annotations are used
        t_before_mask_load = time.time() if ENABLE_TIMING else None
        mask = None
        if self.use_annotations:
            # Masks are stored as rve archives: <series_id>.1.0.tar.lz4
            mask_path = os.path.join(self.rve_masks_dir, f"{series_id}.1.0.tar.lz4")
            if os.path.exists(mask_path):
                mask = rve.load_sample(mask_path, use_hardware_acceleration=False)

            else:
                # Create empty mask if not found
                mask = torch.zeros_like(volume)
        else:
            # Create empty mask if not using annotations
            mask = torch.zeros_like(volume)
        if ENABLE_TIMING:
            t_after_mask_load = time.time()
            timing_print(f"[TIMING]   - mask handling: {t_after_mask_load - t_before_mask_load:.4f}s")

        t_before_reshape = time.time() if ENABLE_TIMING else None
        # Ensure volume and mask have the right shape (C, D, H, W)
        if len(volume.shape) == 3:  # (D, H, W)
            volume = volume.unsqueeze(0)  # Add channel dimension
        if len(mask.shape) == 3:  # (D, H, W)
            mask = mask.unsqueeze(0)  # Add channel dimension

        # Pad the volume and mask to the desired number of images
        D, H, W = volume.shape[1:]
        if D < self.num_images:
            pad_total = self.num_images - D
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left  # Handles odd padding amounts
            # F.pad pads dimensions in reverse order: (W_left, W_right, H_top, H_bottom, D_front, D_back)
            volume = F.pad(volume, (0, 0, 0, 0, pad_left, pad_right))
            mask = F.pad(mask, (0, 0, 0, 0, pad_left, pad_right))
        elif self.num_images < D:
            crop_total = D - self.num_images
            crop_left = crop_total // 2
            crop_right = crop_total - crop_left  # Handles odd padding amounts
            volume = volume[:, crop_left:-crop_right, :, :]
            mask = mask[:, crop_left:-crop_right, :, :]

        crop_side = 0
        if H > 256:
            crop_side = (H - 256) // 2

            volume = volume[:, :, crop_side:-crop_side, crop_side:-crop_side]
            mask = mask[:, :, crop_side:-crop_side, crop_side:-crop_side]
        if ENABLE_TIMING:
            t_after_reshape = time.time()
            timing_print(f"[TIMING]   - reshape/pad/crop: {t_after_reshape - t_before_reshape:.4f}s")

        t_before_float = time.time() if ENABLE_TIMING else None
        # Return a fresh dict to avoid mutating and retaining large tensors on the dataset sample
        result = {
            "input": volume.float(),
            "mask": mask.float(),
        }
        if ENABLE_TIMING:
            t_after_float = time.time()
            timing_print(f"[TIMING]   - .float() conversion: {t_after_float - t_before_float:.4f}s")
            timing_print(f"[TIMING]   - Total read_resample_volume_from_disk: {t_after_float - t_start:.4f}s")

        return result

    def get_images(self, paths, sample):
        """
        RVE-only loading path. Always load volume and mask with rve.load_sample; no fallback.
        """
        if self.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        # Enforce using RVE cache only
        assert not self.resample_input, "RVECacheNLST must not use resample_input path"

        out_dict = self.read_resample_volume_from_disk(paths, sample)

        return out_dict
