import torch
import math
import numpy as np
from postprocess import get_bboxes, box_iou_xyxy, batch_box_iou
from torchmetrics.detection import MeanAveragePrecision
from utils import xywh_to_xyxy
from tqdm import tqdm

C = 13



def batch_box_iou(boxes1, boxes2):
    """
    IoU between two sets of boxes in xyxy format.
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)



class DetectionMetric:
    def box_iou(self, box1, box2, xyxy=True, CIoU=False):
        """
        Intersection Over Union
        Args:
            box1 (tensor): bounding box 1
            box2 (tensor): bounding box 2
            xyxy (bool): whether format is x1y1x2y2
            CIoU (bool): if True calculate CIoU else IoU
        """
        if xyxy:
            box1_x1 = box1[..., 0:1]
            box1_y1 = box1[..., 1:2]
            box1_x2 = box1[..., 2:3]
            box1_y2 = box1[..., 3:4]
            box2_x1 = box2[..., 0:1]
            box2_y1 = box2[..., 1:2]
            box2_x2 = box2[..., 2:3]
            box2_y2 = box2[..., 3:4]
        else:
            box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
            box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
            box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
            box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
            box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
            box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
            box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
            box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_width = box1_x2 - box1_x1
        box1_height = box1_y2 - box1_y1
        box2_width = box2_x2 - box2_x1
        box2_height = box2_y2 - box2_y1
        union = box1_width * box1_height + box2_width * box2_height - intersection
        iou = intersection / (union + 1e-6)

        if CIoU:
            convex_width = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
            convex_height = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
            convex_diag_sq = convex_width**2 + convex_height**2
            center_dist_sq = (
                (box2_x1 + box2_x2 - box1_x1 - box1_x2)**2 +
                (box2_y1 + box2_y2 - box1_y1 - box1_y2)**2
            )
            dist_penalty = center_dist_sq / convex_diag_sq
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(box2_width / box2_height) - torch.atan(box1_width / box1_height), 2
            )
            with torch.no_grad():
                alpha = v / ((1 + 1e-6) - iou + v)
            return iou - dist_penalty + alpha * v

        return iou

    def compute_map(self, predictions, targets, img_height, img_width,
                    iou_threshold=0.5, conf_threshold=0.25, n_classes=C):
        pred_boxes = {c: [] for c in range(n_classes)}
        gt_counts = {c: 0 for c in range(n_classes)}

        for batch_preds, batch_targets in tqdm(zip(predictions, targets), total=len(predictions), desc='Computing mAP'):
            det_results = get_bboxes(batch_preds, img_height, img_width,
                                     iou_threshold=iou_threshold,
                                     conf_threshold=conf_threshold)
            batch_size = batch_targets[0].shape[0]

            for b in range(batch_size):
                gt_by_class = {c: [] for c in range(n_classes)}
                for scale_target in batch_targets:
                    t = scale_target[b]
                    Iobj = t[..., C] == 1
                    if not Iobj.any():
                        continue
                    obj_entries = t[Iobj]
                    classes = obj_entries[..., :C].argmax(dim=-1)
                    boxes = obj_entries[..., C+1:C+5].clone()
                    boxes[..., 0] *= img_width
                    boxes[..., 1] *= img_height
                    boxes[..., 2] *= img_width
                    boxes[..., 3] *= img_height
                    boxes_xyxy = xywh_to_xyxy(boxes)
                    for cls, box in zip(classes.tolist(), boxes_xyxy):
                        gt_by_class[int(cls)].append(box)
                        gt_counts[int(cls)] += 1

                dets = det_results[b]
                if dets.shape[0] == 0:
                    continue

                for cls in range(n_classes):
                    cls_mask = dets[:, 0] == cls
                    cls_dets = dets[cls_mask]
                    if cls_dets.shape[0] == 0:
                        continue
                    cls_dets = cls_dets[cls_dets[:, 1].argsort(descending=True)]

                    if len(gt_by_class[cls]) == 0:
                        for d in cls_dets:
                            pred_boxes[cls].append((d[1].item(), 0))
                        continue

                    gt_boxes = torch.stack(gt_by_class[cls])
                    det_boxes_xyxy = xywh_to_xyxy(cls_dets[:, 2:6])
                    ious = batch_box_iou(det_boxes_xyxy, gt_boxes)
                    best_ious, best_idxs = ious.max(dim=1)

                    matched = [False] * len(gt_by_class[cls])
                    for det_i in range(cls_dets.shape[0]):
                        conf = cls_dets[det_i, 1].item()
                        if best_ious[det_i].item() >= iou_threshold and not matched[best_idxs[det_i].item()]:
                            pred_boxes[cls].append((conf, 1))
                            matched[best_idxs[det_i].item()] = True
                        else:
                            pred_boxes[cls].append((conf, 0))

        per_class_ap = {}
        for cls in range(n_classes):
            if gt_counts[cls] == 0:
                continue
            entries = sorted(pred_boxes[cls], key=lambda x: -x[0])
            tp_cumsum = np.cumsum([e[1] for e in entries])
            fp_cumsum = np.cumsum([1 - e[1] for e in entries])
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recalls = tp_cumsum / (gt_counts[cls] + 1e-6)
            precisions = np.concatenate([[1], precisions])
            recalls = np.concatenate([[0], recalls])
            per_class_ap[cls] = np.trapezoid(precisions, recalls)

        mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        return mAP, per_class_ap