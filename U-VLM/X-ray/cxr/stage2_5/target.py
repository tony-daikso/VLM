"""
Box list -> CenterNet 風格的 heatmap/size/offset/reg_mask target 編碼，dataset.py/train.py/eval.py
共用（見 STAGE2_5_PLAN.md 第 2 節）。`gaussian_radius`/`gaussian2d`/`draw_gaussian` 是 CenterNet
論文的標準做法，直接照搬，不重新發明。
"""
import math

import numpy as np


def gaussian_radius(height, width, min_overlap=0.7):
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / 2

    return max(0.0, min(r1, r2, r3))


def gaussian2d(shape, sigma):
    m, n = [(s - 1.0) / 2.0 for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center_x, center_y, radius):
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x, radius + 1)
    top, bottom = min(center_y, radius), min(height - center_y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    masked_heatmap = heatmap[center_y - top:center_y + bottom, center_x - left:center_x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def encode_targets(boxes, class_index_lists, feature_size, num_classes, min_overlap):
    """
    boxes: list of (x1,y1,x2,y2)，0~1 normalized（增強後的座標）
    class_index_lists: 跟 boxes 等長，每個元素是這個 box 對應的 label_group index 列表（可能 >1 個）
    回傳 heatmap (num_classes, F, F)、size_target (2, F, F)、offset_target (2, F, F)、
    reg_mask (1, F, F) 四個 float32 array，F=feature_size。
    """
    F = feature_size
    heatmap = np.zeros((num_classes, F, F), dtype=np.float32)
    size_target = np.zeros((2, F, F), dtype=np.float32)
    offset_target = np.zeros((2, F, F), dtype=np.float32)
    reg_mask = np.zeros((1, F, F), dtype=np.float32)

    for (x1, y1, x2, y2), class_indices in zip(boxes, class_index_lists):
        w = (x2 - x1) * F
        h = (y2 - y1) * F
        if w <= 0 or h <= 0:
            continue
        cx = (x1 + x2) / 2 * F
        cy = (y1 + y2) / 2 * F
        cx_int, cy_int = int(cx), int(cy)
        cx_int = min(max(cx_int, 0), F - 1)
        cy_int = min(max(cy_int, 0), F - 1)

        radius = max(0, int(round(gaussian_radius(h, w, min_overlap))))
        for cls in class_indices:
            draw_gaussian(heatmap[cls], cx_int, cy_int, radius)

        size_target[:, cy_int, cx_int] = [w, h]
        offset_target[:, cy_int, cx_int] = [cx - cx_int, cy - cy_int]
        reg_mask[0, cy_int, cx_int] = 1.0

    return heatmap, size_target, offset_target, reg_mask
