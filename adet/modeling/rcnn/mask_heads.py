# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Seunghyeok Back (GIST), 2021.

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List
import copy
import pycocotools.mask as mask_utils

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures.masks import PolygonMasks
from detectron2.structures import Instances
from skimage.morphology import  binary_dilation, binary_erosion, square


from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
from abc import abstractmethod

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

def get_crop_bitmask_areas(bitmasks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    
    results = [
       torch.sum(_crop_bitmask(mask, box))for mask, box in zip(bitmasks, boxes)
    ]
    return torch.Tensor(results)

def _crop_bitmask(bitmask: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    x1 = int(box[0])
    x2 = int(box[2])
    y1 = int(box[1])
    y2 = int(box[3])
    return bitmask[x1:x2, y1:y2]

def get_bitmask_areas(bitmasks:  torch.Tensor) -> torch.Tensor:
    return torch.tensor([torch.sum(mask) for mask in bitmasks])

def compute_dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return torch.mean(1 - d)

def Max(x):
    """
    A wrapper around torch.max in Spatial Attention Module (SAM) to support empty inputs and more features.
    """
    if x.numel() == 0:
        output_shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        empty = _NewEmptyTensorOp.apply(x, output_shape)
        return empty
    return torch.max(x, dim=1, keepdim=True)[0]

def get_instances_contour(instances_mask):
    instances_mask = instances_mask.data
    result_c = np.zeros_like(instances_mask, dtype=np.uint8)
    contour = get_contour(instances_mask)
    result_c = np.maximum(result_c, contour)
    contour = np.where(contour > 0, 1, 0)
    return result_c

def get_contour(mask):

    outer = binary_dilation(mask) #, square(9))
    inner = binary_erosion(mask) #, square(9))
    contour = ((outer != inner) > 0).astype(np.uint8)

    return contour

@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], target="gt_mask", dice_loss=False):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.get(target).crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    if dice_loss:
        mask_loss += compute_dice_loss(pred_mask_logits.sigmoid(), gt_masks)
    return mask_loss


@torch.jit.unused
def visible_mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("visible_mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("visible_mask_rcnn/false_positive", false_positive)
    storage.put_scalar("visible_mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss

@torch.jit.unused
def occlusion_mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_occluded_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("occlusion_mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("occlusion_mask_rcnn/false_positive", false_positive)
    storage.put_scalar("occlusion_mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def occlusion_classifcation_loss(occlusion_logits, instances):

    gt_occlusion_bool = cat([p.gt_occluded_rate >= 0.05 for p in instances], dim=0)
    return F.cross_entropy(occlusion_logits, gt_occlusion_bool.to(torch.int64), reduction="mean", weight=torch.Tensor([1, 8]).to(gt_occlusion_bool.device))
      
def occlusion_classifcation_inference(occlusion_logits, instances):

    gt_occlusion = cat([p.gt_occluded_rate >= 0.05 for p in instances], dim=0)
    return F.cross_entropy(occlusion_logits, gt_occlusion, reduction="mean")

def mask_rcnn_inference(pred_mask_logits, pred_instances, target="pred_masks"):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.set(target, prob)  # (1, Hmask, Wmask)

def visible_mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_visible_masks = prob  # (1, Hmask, Wmask)

def occlusion_mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_occlusion_masks = prob  # (1, Hmask, Wmask)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_MASK_HEAD_REGISTRY.register()
class VisibleMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(VisibleMaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        self.input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self.input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("visible_mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)


        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else self.input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        
        self.mlc_layers = []
        self.mlc_layers.append(Conv2d(2*conv_dims, 2*conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2*conv_dims, 2*conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2*conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
        for i, layer in enumerate(self.mlc_layers):
            self.add_module("visible_mlc_layer{}".format(i), layer)

        self.guide_conv_layers = []
        n_features = self.prediction_order.index("V") + 1 if not cfg.MODEL.NO_DENSE_GUIDANCE else 1
        if self.hom and self.guidance_type == "concat":
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, n_features*conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, n_features*conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.guide_conv_layers):
                self.add_module("visible_guidance_layer{}".format(i), layer)


        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv, self.predictor] + self.mlc_layers + self.guide_conv_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x, instances, extracted_features):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances: contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
            vis_mask_features: visible mask features from visible mask head
        Returns:
            mask_logits: predicted mask logits from given region features
        """
        if self.hom and self.guidance_type == "concat":
            for layer in self.guide_conv_layers:
                x = layer(x)

        for i, layer in enumerate(self.conv_norm_relus):
            if i == 0 and self.MLC:
                x = layer(x)
                x = torch.cat([x, extracted_features], 1)
                for mlc_layer in self.mlc_layers:
                    x = mlc_layer(x)
            else:
                x = layer(x)
        mask_logits = self.predictor(F.relu(self.deconv(x)))
        return mask_logits, x

@ROI_MASK_HEAD_REGISTRY.register()
class AmodalMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(AmodalMaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        self.input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER
        
        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                self.input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("amodal_mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)


        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else self.input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.mlc_layers = []
        self.mlc_layers.append(Conv2d(2*conv_dims, 2*conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2*conv_dims, 2*conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2*conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
        for i, layer in enumerate(self.mlc_layers):
            self.add_module("amodal_mlc_layer{}".format(i), layer)
            
        self.guide_conv_layers = []
        n_features = self.prediction_order.index("A") + 1 if not cfg.MODEL.NO_DENSE_GUIDANCE else 2
        if self.hom and self.guidance_type == "concat":
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, n_features*conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, n_features*conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.guide_conv_layers):
                self.add_module("amodal_guidance_layer{}".format(i), layer)
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv, self.predictor] + self.mlc_layers + self.guide_conv_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x, instances, extracted_features):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances: contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
            vis_mask_features: visible mask features from visible mask head
        Returns:
            mask_logits: predicted mask logits from given region features
        """
        if self.hom and self.guidance_type == "concat":
            for layer in self.guide_conv_layers:
                x = layer(x)

        for i, layer in enumerate(self.conv_norm_relus):
            if i == 0 and self.MLC:
                x = layer(x)
                x = torch.cat([x, extracted_features], 1)
                for mlc_layer in self.mlc_layers:
                    x = mlc_layer(x)
            else:
                x = layer(x)
        mask_logits = self.predictor(F.relu(self.deconv(x)))
        return mask_logits, x


class OCCCLSMaskHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, name=""):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(OCCCLSMaskHead, self).__init__()

        # fmt: off
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        # fmt: on
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER
        self.conv_norm_relus = []
        
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=2 if k==1 else 1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("{}_occ_cls_fcn{}".format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)
        
        self.mlc_layers = []
        self.mlc_layers.append(Conv2d(2*conv_dims, 2*conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2*conv_dims, 2*conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2*conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
        for i, layer in enumerate(self.mlc_layers):
            self.add_module("occ_cls_{}_mlc_layer{}".format(name, i), layer)
        
        self.guide_conv_layers = []
        n_features = self.prediction_order.index("O") + 1 if not cfg.MODEL.NO_DENSE_GUIDANCE else 2
        if self.hom and self.guidance_type == "concat":
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, n_features*conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, n_features*conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features*conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.guide_conv_layers):
                self.add_module("occlusion_guidance_layer{}".format(i), layer)
        
        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        input_size = input_shape.channels * (input_shape.width//2 or 1) * (input_shape.height//2 or 1)
        self.predictor = nn.Linear(input_size, 2)
        weight_init.c2_xavier_fill(self.predictor)
        
        for layer in self.conv_norm_relus + self.mlc_layers + [self.predictor, self.deconv] + self.guide_conv_layers:
            weight_init.c2_msra_fill(layer)


    def forward(self, x, instances, extracted_features):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances: contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            mask_logits: predicted mask logits from given region features
        """
        
        if self.hom and self.guidance_type == "concat":
            for layer in self.guide_conv_layers:
                x = layer(x)
        
        for i, layer in enumerate(self.conv_norm_relus):
            x = layer(x)
            if i == 0 and self.MLC:
                x = torch.cat([x, extracted_features], 1)
                for mlc_layer in self.mlc_layers:
                    x = mlc_layer(x)
        if x.dim() > 2:
            x_flatten = torch.flatten(x, start_dim=1)
        class_logits = self.predictor(x_flatten)
        return class_logits, self.deconv(x)

def build_amodal_mask_head(cfg, input_shape):
    """
    Build a amodal mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)

def build_visible_mask_head(cfg, input_shape):
    name = cfg.MODEL.ROI_VISIBLE_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)

def build_occlusion_classification_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_OCC_CLS_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_OCC_CLS_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)