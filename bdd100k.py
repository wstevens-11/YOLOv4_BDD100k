import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from torchvision.io import read_image

from utils import xyxy_to_xywh

ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)]
]

CLASS_DICT = {
    'pedestrian':    1,
    'rider':         2,
    'car':           3,
    'truck':         4,
    'bus':           5,
    'train':         6,
    'motorcycle':    7,
    'bicycle':       8,
    'traffic light': 9,
    'traffic sign':  10,
    'other vehicle': 11,
    'trailer':       12,
    'other person':  13,
}


class BDD100k(data.Dataset):
    """
    Detection-only dataloader for the BDD100K dataset.
    - root: path to dataset root
    - train: boolean to load train or val split
    - transform: transform applied to images
    - S: grid sizes for each detection scale
    - anchors: anchor boxes for each scale
    - timeofday: filter by time of day ('daytime', 'night', 'dawn/dusk', or None)
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        S=[(90, 160), (45, 80), (23, 40)],
        anchors=ANCHORS,
        timeofday=None,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.class_dict = CLASS_DICT
        self.S = S
        self.C = len(self.class_dict)

        assert len(anchors) == len(S), "Number of anchor groups must match number of scales"

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.n_anchors = self.anchors.shape[0]
        self.n_anchors_scale = self.n_anchors // 3
        self.ignore_iou_thresh = 0.5

        # Load annotations
        label_file = 'det_v2_train_release.json' if train else 'det_v2_val_release.json'
        label_path = os.path.join(root, 'labels', label_file)
        self.detect = pd.read_json(label_path)

        # Drop images with no detection labels
        self.detect.dropna(axis=0, subset=['labels'], inplace=True)
        self.detect.reset_index(drop=True, inplace=True)

        # Filter by time of day (e.g. 'daytime', 'night', 'dawn/dusk')
        if timeofday is not None:
            mask = self.detect['attributes'].apply(
                lambda a: a.get('timeofday') == timeofday if isinstance(a, dict) else False
            )
            self.detect = self.detect[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.detect)

    def _iou_anchors(self, box, anchors):
        """
        Compute IoU between a single box (w, h) and all anchor boxes
        """
        intersection = torch.min(box[0], anchors[:, 0]) * torch.min(box[1], anchors[:, 1])
        union = (box[0] * box[1] + anchors[:, 0] * anchors[:, 1]) - intersection
        return intersection / union

    def __getitem__(self, index):
        target = self.detect.iloc[index]
        split = 'train' if self.train else 'val'
        img_path = os.path.join(self.root, 'images', '100k', split, target['name'])
        img = read_image(img_path).float()
        _, height, width = img.shape

        if self.transform:
            img = self.transform(img)

        label = [
            torch.zeros(self.n_anchors_scale, Sy, Sx, self.C + 5)
            for Sy, Sx in self.S
        ]

        annotations = target['labels']
        for obj in annotations:
            category = obj.get('category')
            if category not in self.class_dict:
                continue  # skip unlabeled or unknown categories
            if obj.get('box2d') is None:
                continue  # skip objects without a 2D box

            obj_class = self.class_dict[category]
            box2d = obj['box2d']
            bbox_xyxy = np.array([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)

            # Normalize coordinates to [0, 1]
            x  = bbox_xywh[0] / width
            y  = bbox_xywh[1] / height
            w  = bbox_xywh[2] / width
            h  = bbox_xywh[3] / height

            # Find best matching anchor
            box_wh = torch.tensor([w, h])
            anchors_iou = self._iou_anchors(box_wh, self.anchors)
            anchor_idx = torch.argmax(anchors_iou).item()

            scale_idx = anchor_idx // self.n_anchors_scale
            anchor    = anchor_idx %  self.n_anchors_scale
            Sy, Sx    = self.S[scale_idx]

            i = int(Sy * y)  # grid row
            j = int(Sx * x)  # grid col

            # Skip if an anchor is already assigned at this cell
            if label[scale_idx][anchor, i, j, self.C] == 1:
                continue

            label[scale_idx][anchor, i, j, self.C] = 1
            label[scale_idx][anchor, i, j, self.C + 1] = x      # absolute normalized x
            label[scale_idx][anchor, i, j, self.C + 2] = y      # absolute normalized y
            label[scale_idx][anchor, i, j, self.C + 3] = w      # absolute normalized w
            label[scale_idx][anchor, i, j, self.C + 4] = h      # absolute normalized h
            label[scale_idx][anchor, i, j, obj_class - 1] = 1

            # Ignore lower-IoU anchors
            for a_idx in range(self.n_anchors):
                if a_idx == anchor_idx:
                    continue
                s_idx = a_idx // self.n_anchors_scale
                a = a_idx %  self.n_anchors_scale
                Sy2, Sx2 = self.S[s_idx]
                i2 = int(Sy2 * y)
                j2 = int(Sx2 * x)
                if (label[s_idx][a, i2, j2, self.C] != 1
                        and anchors_iou[a_idx] > self.ignore_iou_thresh):
                    label[s_idx][a, i2, j2, self.C] = -1

        return img / 255.0, label