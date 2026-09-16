"""
heatmap/size/offset -> box 解碼（3x3 max-pool 找 peak，等同一種簡化版 NMS）+ 簡化版單一 IoU 門檻
per-class AP，train.py（算 val mAP 當 early-stopping 指標）/ eval.py（完整報告）共用，見
STAGE2_5_PLAN.md 第 6 節。
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_predictions(heatmap, size, offset, topk):
    """heatmap/size/offset: (B, C/2/2, Fh, Fw) tensor（heatmap 已經 sigmoid 過）。
    回傳長度 B 的 list，每個元素是 list of (score, class_idx, x1, y1, x2, y2)，座標是
    0~1 normalized（已經用 feature map 大小換算回去、clip 到 [0,1]）。"""
    Bn, C, Fh, Fw = heatmap.shape
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = heatmap * (hmax == heatmap).float()

    scores, indices = torch.topk(peaks.view(Bn, -1), min(topk, C * Fh * Fw))
    classes = indices // (Fh * Fw)
    rem = indices % (Fh * Fw)
    ys = rem // Fw
    xs = rem % Fw

    results = []
    for b in range(Bn):
        dets = []
        for k in range(scores.shape[1]):
            score = scores[b, k].item()
            if score <= 0:
                continue
            cls = classes[b, k].item()
            x, y = xs[b, k].item(), ys[b, k].item()
            dx, dy = offset[b, 0, y, x].item(), offset[b, 1, y, x].item()
            w, h = size[b, 0, y, x].item(), size[b, 1, y, x].item()
            cx, cy = x + dx, y + dy
            x1, y1 = (cx - w / 2) / Fw, (cy - h / 2) / Fh
            x2, y2 = (cx + w / 2) / Fw, (cy + h / 2) / Fh
            x1, x2 = np.clip([x1, x2], 0.0, 1.0)
            y1, y2 = np.clip([y1, y2], 0.0, 1.0)
            dets.append((score, cls, x1, y1, x2, y2))
        results.append(dets)
    return results


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_for_class(gt_boxes_by_item, pred_boxes_by_item, iou_threshold):
    """gt_boxes_by_item: {item_id: [box, ...]}；pred_boxes_by_item: {item_id: [(score, box), ...]}。
    回傳 (ap, n_gt)；n_gt=0 時 ap=None（這個類別在這個 split 沒有 GT box，AP 沒有意義）。"""
    n_gt = sum(len(boxes) for boxes in gt_boxes_by_item.values())
    if n_gt == 0:
        return None, 0

    all_preds = []
    for item_id, preds in pred_boxes_by_item.items():
        for score, box in preds:
            all_preds.append((score, item_id, box))
    all_preds.sort(key=lambda x: -x[0])

    matched = {item_id: [False] * len(boxes) for item_id, boxes in gt_boxes_by_item.items()}
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    for i, (_, item_id, box) in enumerate(all_preds):
        gts = gt_boxes_by_item.get(item_id, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gts):
            if matched[item_id][j]:
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            matched[item_id][best_j] = True
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _voc_ap(recall, precision), n_gt
