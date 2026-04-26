import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes
from utils import xywh_to_xyxy

# Anchors in pixel space at 384x640 (matching what was used for label assignment)
ANCHORS_PX = [
    [(142, 110), (192, 243), (459, 401)],
    [(36, 75), (76, 55), (72, 146)],
    [(12, 16), (19, 36), (40, 28)],
]

C = 13


def get_normalized_anchors(img_width, img_height):
    """Return anchors normalized to [0,1] image space."""
    return [
        [(w / img_width, h / img_height) for w, h in scale]
        for scale in ANCHORS_PX
    ]


def process_predictions(predictions, img_height, img_width):
    """
    Decode raw model outputs into per-image detection lists in pixel coords.
    """
    norm_anchors = get_normalized_anchors(img_width, img_height)
    batch_size = predictions[0].shape[0]
    all_detections = [[] for _ in range(batch_size)]

    for i, prediction in enumerate(predictions):
        anchor = torch.tensor(norm_anchors[i], dtype=torch.float32).view(1, 3, 1, 1, 2)
        pred = prediction.clone().detach().cpu()
        B, n_anchors, gy, gx, n_outputs = pred.shape

        gridy, gridx = torch.meshgrid(torch.arange(gy), torch.arange(gx), indexing='ij')
        gridx = gridx.view(1, 1, gy, gx).float()
        gridy = gridy.view(1, 1, gy, gx).float()

        pred[..., C+1] = pred[..., C+1].sigmoid()
        pred[..., C+2] = pred[..., C+2].sigmoid()
        pred[..., C+3:C+5] = pred[..., C+3:C+5].clamp(max=4).exp()

        confidence = pred[..., C].sigmoid()
        predicted_class = torch.argmax(pred[..., :C], dim=-1).float()

        # x, y in normalized [0,1] then to pixels
        x_norm = (pred[..., C+1] + gridx) / gx
        y_norm = (pred[..., C+2] + gridy) / gy
        # w, h in normalized [0,1] then to pixels
        w_norm = pred[..., C+3] * anchor[..., 0]
        h_norm = pred[..., C+4] * anchor[..., 1]

        x = x_norm * img_width
        y = y_norm * img_height
        w = w_norm * img_width
        h = h_norm * img_height

        det = torch.stack([predicted_class, confidence, x, y, w, h], dim=-1)
        det = det.view(B, -1, 6)
        for b in range(B):
            all_detections[b].append(det[b])

    return [torch.cat(dets, dim=0) for dets in all_detections]


def nms(detections, iou_threshold=0.5, conf_threshold=0.25):
    mask = detections[:, 1] > conf_threshold
    detections = detections[mask]
    if detections.shape[0] == 0:
        return detections

    detections = detections[detections[:, 1].argsort(descending=True)]
    kept = []
    while detections.shape[0] > 0:
        best = detections[0]
        kept.append(best)
        if detections.shape[0] == 1:
            break
        rest = detections[1:]
        best_box = xywh_to_xyxy(best[2:6].unsqueeze(0))
        rest_boxes = xywh_to_xyxy(rest[:, 2:6])
        ious = box_iou_xyxy(best_box, rest_boxes)
        same_class = rest[:, 0] == best[0]
        suppress = same_class & (ious.squeeze(0) >= iou_threshold)
        detections = rest[~suppress]
    return torch.stack(kept)


def box_iou_xyxy(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def batch_box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def get_bboxes(predictions, img_height, img_width, iou_threshold=0.5, conf_threshold=0.25):
    all_detections = process_predictions(predictions, img_height, img_width)
    return [nms(dets, iou_threshold, conf_threshold) for dets in all_detections]


def draw_bbox(img, detections, class_names):
    if detections.shape[0] == 0:
        return img
    img = img.to(torch.uint8).cpu()
    boxes = xywh_to_xyxy(detections[:, 2:6])
    labels = [f"{class_names.get(int(d[0].item()), '?')} {d[1]:.2f}" for d in detections]
    return draw_bounding_boxes(img, boxes, labels=labels, width=2)