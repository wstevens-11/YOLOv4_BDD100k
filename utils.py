import torch
import torch.nn as nn
import numpy as np

def bbox_iou(pred:torch.Tensor, target: torch.Tensor, format:str = 'xyxy'):
    '''
    Caculate IOU between 2 boxes
    - pred: predictions tensor
    - target: target tensor
    - format: xyxy or xywh
    '''
    assert pred.shape[0] == target.shape[0]
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    
def xyxy_to_xywh(x):
    '''
    Convert x1, y1, x2, y2 format to xywh format
    - x: tuple of two bounding box coordinates (top left, bottom right corners)
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y
def xywh_to_xyxy(x):
    '''
    Convert x1, y1, w, h format to xyxt format
    - x: tuple of bounding box coordinates (top left point, width, height)
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y
def norm_bbox(x, height, width):
    '''
    Normalize bounding box coordinates to given height and width
    - x: tuple of bounding box coordinates (top left point, width, height)
    - height: height normalizer
    - width: width normalizer
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] /= width
    y[..., 1] /= height
    y[..., 2] /= width
    y[..., 3] /= height
    return y
def unnorm_bbox(x, height, width):
    '''
    Un-normalize bounding box coordinates away from given height and width normalizers
    - x: tuple of bounding box coordinates (top left point, width, height)
    - height: height normalizer
    - width: width normalizer
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] *= width
    y[..., 1] *= height
    y[..., 2] *= width
    y[..., 3] *= height