import albumentations as A
import torch
import cv2

import kornia.augmentation as K
from pillar.augmentations.abstract import AbstractAugmentation


class Scale2d(AbstractAugmentation):
    """
    Given PIL image, enforce its some set size
    (can use for down sampling / keep full res)
    """

    def __init__(self, args, interpolation="cubic", img_size=None, **extras):
        super(Scale2d, self).__init__(args)
        interpolation_map = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
        interpolation = interpolation_map[interpolation]
        if img_size is not None:
            height, width = img_size
        else:
            height, width = args.dataset.img_size
        self.set_cachable(height, width, interpolation)
        self.transform = A.Resize(height, width, interpolation=interpolation)

    def __call__(self, img, **extras):
        if isinstance(img, dict):
            transform_output = self.transform(image=img["input"], mask=img["mask"])
            img["input"], img["mask"] = (
                transform_output["image"],
                transform_output["mask"],
            )
            return img

        return self.transform(image=img)["image"]


class RotateRange(AbstractAugmentation):
    """
    Rotate image counter clockwise by random
    kwargs['limit'] degrees.

    Example: 'rotate/limit=20' will rotate by up to +/-20 deg
    """

    def __init__(self, args, limit=10, **extras):
        super(RotateRange, self).__init__(args)
        self.transform = A.Rotate(limit=limit, p=0.5)

    def __call__(self, img, seed=None, **extras):
        if seed:
            self.set_seed(seed)

        if isinstance(img, dict):
            transform_output = self.transform(image=img["input"], mask=img["mask"])
            img["input"], img["mask"] = (
                transform_output["image"],
                transform_output["mask"],
            )
            return img

        return self.transform(image=img)["image"]


class Resize3d(AbstractAugmentation):
    def __init__(self, args, h=None, w=None, d=None, interpolation="trilinear", **extras):
        super().__init__(args)
        self.img_size = (h, w, d)
        self.interpolation = interpolation
        self.set_cachable(h, w, d, interpolation)

    def transform(self, img):
        return torch.nn.functional.interpolate(img, size=self.img_size, mode=self.interpolation)

    def __call__(self, img, **extras):
        return self.transform(img)


class RandomRotation3D(AbstractAugmentation):
    def __init__(self, args, degrees=[0, 0, 10], resample="bilinear", same_on_batch=False, p=0.5, **kwargs):
        super(RandomRotation3D, self).__init__(args)
        self.transform = K.RandomRotation3D(degrees=degrees, resample=resample, same_on_batch=same_on_batch, p=p)

    def __call__(self, img, seed=None, **extras):
        if seed:
            self.set_seed(seed)
        if isinstance(img, dict):
            img["input"] = self.transform(img["input"])
            if "mask" in img and img["mask"] is not None:
                img["mask"] = self.transform(img["mask"])
            return img
        return self.transform(img)


class RandomHorizontalFlip3D(AbstractAugmentation):
    def __init__(self, args, same_on_batch=False, p=0.5, **kwargs):
        super(RandomHorizontalFlip3D, self).__init__(args)
        self.transform = K.RandomHorizontalFlip3D(same_on_batch=same_on_batch, p=p)

    def __call__(self, img, seed=None, **extras):
        if seed:
            self.set_seed(seed)
        if isinstance(img, dict):
            img["input"] = self.transform(img["input"])
            if "mask" in img and img["mask"] is not None:
                img["mask"] = self.transform(img["mask"])
            return img
        return self.transform(img)
