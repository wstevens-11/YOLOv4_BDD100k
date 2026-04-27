import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import DetectionMetric

ANCHORS_PX = [
    [(142, 110), (192, 243), (459, 401)],
    [(36, 75), (76, 55), (72, 146)],
    [(12, 16), (19, 36), (40, 28)],
]


class DetectionLoss(nn.Module):
    def __init__(self, n_classes=13, img_height=384, img_width=640,
                 alpha_class=1., alpha_box=1., alpha_obj=1.):
        super().__init__()
        self.metric = DetectionMetric()
        self.C = n_classes
        self.img_h = img_height
        self.img_w = img_width
        self.alpha_class = alpha_class
        self.alpha_box = alpha_box
        self.alpha_obj = alpha_obj
        # Normalize anchors to [0,1] image space
        self.anchors_norm = [
            [(w / img_width, h / img_height) for w, h in scale]
            for scale in ANCHORS_PX
        ]

    def forward(self, preds, targets):
        ciou_loss = 0.
        obj_loss = 0.
        noobj_loss = 0.
        class_loss = 0.

        for i, pred in enumerate(preds):
            target = targets[i].to(pred.device)
            Sy, Sx = pred.shape[2], pred.shape[3]

            anchor = torch.tensor(self.anchors_norm[i]).view(1, 3, 1, 1, 2).to(pred.device)
            gridx = torch.arange(Sx).view(1, 1, 1, Sx).to(pred.device).float()
            gridy = torch.arange(Sy).view(1, 1, Sy, 1).to(pred.device).float()

            # Encode targets to match raw prediction format
            target_enc = target.clone()
            # x, y: normalized [0,1] -> grid-relative offset [0,1]
            target_enc[..., self.C+1] = target[..., self.C+1] * Sx - gridx
            target_enc[..., self.C+2] = target[..., self.C+2] * Sy - gridy
            # w, h: normalized [0,1] -> log-space ratio relative to anchor
            target_enc[..., self.C+3:self.C+5] = torch.log(
                1e-6 + target[..., self.C+3:self.C+5] / anchor
            )

            Iobj = target[..., self.C] == 1
            Inoobj = target[..., self.C] == 0

            # Apply sigmoid to predicted x, y for CIoU comparison
            pred_for_loss = pred.clone()
            pred_for_loss[..., self.C+1:self.C+3] = pred[..., self.C+1:self.C+3].sigmoid()
            
            pred_for_loss[..., self.C+3:self.C+5] = pred[..., self.C+3:self.C+5].clamp(min=-10, max=10)
            pred_for_loss[..., self.C] = pred[..., self.C].clamp(min=-20, max=20)

            pred_obj = pred_for_loss[Iobj]
            target_obj = target_enc[Iobj]
            pred_noobj = pred_for_loss[Inoobj]
            target_noobj = target_enc[Inoobj]

            if Iobj.any():
                iou = self.metric.box_iou(
                    pred_obj[..., self.C+1:self.C+5],
                    target_obj[..., self.C+1:self.C+5],
                    xyxy=False, CIoU=True
                ).mean()
                ciou_loss += 1 - iou
                class_loss += self._focal_loss(
                    pred_obj[..., :self.C],
                    target_obj[..., :self.C]
                )
                obj_loss += self._focal_loss(
                    pred_obj[..., self.C:self.C+1],
                    target_obj[..., self.C:self.C+1]
                )

            if Inoobj.any():
                noobj_loss += self._focal_loss(
                    pred_noobj[..., self.C:self.C+1],
                    target_noobj[..., self.C:self.C+1]
                )

        return (self.alpha_box * ciou_loss
                + self.alpha_class * class_loss
                + self.alpha_obj * (obj_loss + noobj_loss))

    def _focal_loss(self, preds, targets, alpha=0.25, gamma=2):
        p = torch.sigmoid(preds)
        ce = F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** gamma)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (loss * alpha_t).mean()