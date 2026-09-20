"""
RLE decode helper for the CheXmask Database (PhysioNet
`chexmask-cxr-segmentation-data`, v1.0.0).

`get_mask_from_RLE` is copied verbatim (only renamed for clarity) from the
official CheXmask-Database repo:
https://github.com/ngaggion/CheXmask-Database/blob/main/DataPostprocessing/utils.py

The CSV's Left Lung/Right Lung/Heart columns each hold a run-length encoded
binary mask as a space-separated "start length start length ..." string,
with 1-indexed start positions into a row-major-flattened (height, width)
array.
"""
import numpy as np


def get_mask_from_rle(rle: str, height: int, width: int) -> np.ndarray:
    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        start -= 1
        end = start + length
        mask[start:end] = 1
    return mask.reshape((height, width))
