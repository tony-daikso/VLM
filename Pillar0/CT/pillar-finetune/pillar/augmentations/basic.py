import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF

from pillar.augmentations.abstract import AbstractAugmentation


class ToDtype(AbstractAugmentation):
    """
    Cast the input to the specified dtype.
    """

    def __init__(self, args, dtype="float16", backend="numpy", **extras):
        super().__init__(args)
        self.dtype = dtype
        self.backend = backend

        self.dtype_api_map = {
            "torch_bf16": torch.bfloat16,
            "torch_float16": torch.float16,
            "torch_int16": torch.int16,
            "torch_float32": torch.float32,
            "numpy_bf16": np.float16,
            "numpy_float16": np.float16,
            "numpy_int16": np.int16,
            "numpy_float32": np.float32,
        }

        if dtype in ["bf16", "float16", "int16"]:
            self.set_cachable(f"{backend}_{dtype}")

    def transform(self, img):
        to_dtype = self.dtype_api_map[f"{self.backend}_{self.dtype}"]
        if self.backend == "torch":
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            return img.to(to_dtype)
        if self.backend == "numpy":
            return img.astype(to_dtype)

    def __call__(self, img, **kwargs):
        # mask is fine to keep at float32
        if isinstance(img, dict):
            img["input"] = self.transform(img["input"])
            # img["mask"] = self.transform(img["mask"])
            return img

        return self.transform(img)


class ToTensor(AbstractAugmentation):
    """Converts a numpy array to a torch tensor with the channel dimension last.

    Does not normalize the range of the image.
    """

    def __init__(self, args, normalize_range=False, **extras):
        super().__init__(args)
        self.normalize_range = normalize_range
        self.set_cachable(normalize_range)

    def transform(self, img):
        if self.normalize_range:
            return TF.to_tensor(img).permute(1, 2, 0)
        # Copy if the numpy array is non-writable to prevent warning
        # We can also silence the warning without copying the array (since most of the time the array will not be modified) if required by the latency or memory.
        if img.flags.writeable:
            return torch.from_numpy(img).float()
        else:
            return torch.tensor(img).float()

    def __call__(self, img, **kwargs):
        if isinstance(img, dict):
            img["input"] = self.transform(img["input"])
            img["mask"] = self.transform(img["mask"])

            return img

        return self.transform(img)


class ComposeAug(AbstractAugmentation):
    """
    composes multiple augmentations
    """

    def __init__(self, args, augmentations):
        super(ComposeAug, self).__init__(args)
        self.augmentations = augmentations

    def __call__(self, img, **kwargs):
        for transformer in self.augmentations:
            img = transformer(img, **kwargs)

        return img
