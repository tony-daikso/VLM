import torch
import os
import json
import cv2
import numpy as np
import pydicom
import torchio as tio
import math

from pillar.utils.logging import logger

from .abstract_loader import AbstractLoader, AbstractGroupLoader, DirectCache
from pillar.datasets.nlst_utils import get_scaled_annotation_mask
from pydicom.pixel_data_handlers.util import apply_modality_lut

LOADING_ERROR = "LOADING ERROR! {}"


class DicomLoaderMixin(object):
    def _dicom_loader_init(self):
        self.window_center = -600
        self.window_width = 1500

    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        mask = (
            get_scaled_annotation_mask(sample["annotations"], self.args)
            if self.args.dataset.shared_dataset_kwargs.use_annotations
            else None
        )
        if path == self.pad_token:
            shape = (
                self.args.dataset.num_chan,
                self.args.dataset.img_size[0],
                self.args.dataset.img_size[1],
            )
            arr = torch.zeros(*shape)
            mask = (
                torch.from_numpy(mask * 0).unsqueeze(0)
                if self.args.dataset.shared_dataset_kwargs.use_annotations
                else None
            )
        else:
            try:
                dcm = pydicom.dcmread(path)
                dcm = apply_modality_lut(dcm.pixel_array, dcm)
                arr = apply_windowing(dcm, self.window_center, self.window_width)
                arr = arr // 256  # parity with images loaded as 8 bit
            except Exception:
                raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))

        return {"input": arr, "mask": mask}

    @property
    def cached_extension(self):
        return ""


class DicomLoader(DicomLoaderMixin, AbstractLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dicom_loader_init()


class DicomGroupLoader(DicomLoaderMixin, AbstractGroupLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dicom_loader_init()


class ResampleDicomGroupLoader(DicomGroupLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resample_cache = DirectCache(os.path.join(self.cache.cache_dir, "resample"))

    def write_volume_to_disk(self, paths, sample):
        self.resample_cache.add(paths, sample)

    def apply_random_augs(self, image):
        attr_key, augmentations = self.split_augmentations[0]
        if len(augmentations):
            for idx, transform in enumerate(augmentations):
                image = transform(image)
        return image

    def read_volume_from_disk(self, paths, sample):
        image = self.resample_cache.get(paths, sample)
        # image = self.apply_random_augs(image)
        return image


class TorchIOLoader(AbstractLoader):
    def __init__(self, *args, process_annotations_on_cached=False, image_key=None, **kwargs):
        # Assumption: Annotations are already included in the cached TorchIO subject
        # Will be true if there is a TorchIO.Unpack
        super().__init__(*args, process_annotations_on_cached=process_annotations_on_cached, **kwargs)
        self.image_keys = []
        self.image_key = image_key

    def load_input(self, path, additional=None):
        """
        path here is defined as subject dict, serialized to json str
        """
        subject_dict = json.loads(path)
        self.image_keys = []
        try:
            # Load images
            for k, v in subject_dict["image_path"].items():
                subject_dict[k] = tio.ScalarImage(v)
            for k, v in subject_dict["segmentation_path"].items():
                subject_dict[k] = tio.LabelMap(v)

            subject = tio.Subject(subject_dict)
            subject.load()
        except:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD TorchIo Subject."))
        return subject

    def process_annotations(self, image, additional):
        """
        The annotations are passed in the 'cancer_mask' key of additional.

        Args:
            image: The tio.Subject containing the loaded data
            additional: Dictionary with additional data including cancer_mask
        Returns:
            image: Updated image with annotations if provided
        """
        if additional and "cancer_mask" in additional and additional["cancer_mask"] is not None:
            image_key = self.image_key if self.image_key is not None else next(iter(self.image_keys))
            affine = image[image_key].affine
            image["cancer_mask"] = tio.LabelMap(tensor=additional["cancer_mask"], affine=affine)

        return image

    @property
    def cached_extension(self):
        return ".tio"


def apply_windowing(image, center, width, bit_size=16):
    """
    Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.

    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2

    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460

    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    y_min = 0
    y_max = 2**bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    below = image <= (c - w / 2)  # pixels to be set as black
    above = image > (c + w / 2)  # pixels to be set as white
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image


def get_scaled_annotation_mask(additional, args, scale_annotation=True):
    """
    Construct bounding box masks for annotations.

    Args:
        - additional['image_annotations']: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1].
        - args
    Returns:
        - mask of same size as input image, filled in where bounding box was drawn. If additional['image_annotations'] = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
    """
    H, W = args.dataset.img_size
    mask = np.zeros((H, W))
    if additional["image_annotations"] is None:
        return mask

    for annotation in additional["image_annotations"]:
        single_mask = np.zeros((H, W))
        x_left, y_top = annotation["x"] * W, annotation["y"] * H
        x_right, y_bottom = (
            x_left + annotation["width"] * W,
            y_top + annotation["height"] * H,
        )

        # pixels completely inside bounding box
        x_quant_left, y_quant_top = math.ceil(x_left), math.ceil(y_top)
        x_quant_right, y_quant_bottom = math.floor(x_right), math.floor(y_bottom)

        # excess area along edges
        dx_left = x_quant_left - x_left
        dx_right = x_right - x_quant_right
        dy_top = y_quant_top - y_top
        dy_bottom = y_bottom - y_quant_bottom

        # fill in corners first in case they are over-written later by greater true intersection
        # corners
        single_mask[math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
        single_mask[math.floor(y_top), x_quant_right] = dx_right * dy_top
        single_mask[y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
        single_mask[y_quant_bottom, x_quant_right] = dx_right * dy_bottom

        # edges
        single_mask[y_quant_top:y_quant_bottom, math.floor(x_left)] = dx_left
        single_mask[y_quant_top:y_quant_bottom, x_quant_right] = dx_right
        single_mask[math.floor(y_top), x_quant_left:x_quant_right] = dy_top
        single_mask[y_quant_bottom, x_quant_left:x_quant_right] = dy_bottom

        # completely inside
        single_mask[y_quant_top:y_quant_bottom, x_quant_left:x_quant_right] = 1

        # in case there are multiple boxes, add masks and divide by total later
        mask += single_mask

    if scale_annotation:
        mask /= mask.sum()
    return mask
