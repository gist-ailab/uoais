import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

import pycocotools.mask as mask_utils

def rle_to_bitmask(rles) -> "BitMasks":
        """
        Args:
            rles
            height, width (int)
        """
        masks = [mask_utils.decode(rle).astype(np.uint8) for rle in rles]
        return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

def mask_to_rle(mask):
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def merge_bitmask(masks):
    return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

def transform_segm_in_anno(annotation, transforms, key):
    segm = annotation[key]
    if isinstance(segm, list):
        # polygons
        polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
        annotation[key] = [
            p.reshape(-1) for p in transforms.apply_polygons(polygons)
        ]
    elif isinstance(segm, dict):
        # RLE
        mask = mask_utils.decode(segm)
        mask = transforms.apply_segmentation(mask)
        annotation[key] = mask_to_rle(np.array(mask, dtype=np.uint8, order='F'))
    else:
        raise ValueError(
            "Cannot transform segmentation of type '{}'!"
            "Supported types are: polygons as list[list[float] or ndarray],"
            " COCO-style RLE as a dict.".format(type(segm))
        )
    return annotation

def convert_to_mask(segms):
    masks = []
    for segm in segms:
        if isinstance(segm, list):
            # polygon
            masks.append(polygons_to_bitmask(segm, *image_size))
        elif isinstance(segm, dict):
            # COCO RLE
            masks.append(mask_utils.decode(segm))
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        else:
            raise ValueError(
                "Cannot convert segmentation of type '{}' to BitMasks!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict, or a binary segmentation mask "
                " in a 2D numpy array of shape HxW.".format(type(segm))
            )
    return masks


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        annotation = transform_segm_in_anno(annotation, transforms, "segmentation")

    if "visible_mask" in annotation:
        annotation = transform_segm_in_anno(annotation, transforms, "visible_mask")
        
    if "occluded_mask" in annotation:
        annotation = transform_segm_in_anno(annotation, transforms, "occluded_mask")

        
    return annotation


def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


def annotations_to_instances(annos, image_size, mask_format="polygon", amodal=True):
    
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    occ_classes = [int(obj["occluded_rate"] >= 0.05) for obj in annos]
    target.gt_occludeds = torch.tensor(occ_classes, dtype=torch.int64)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:

        if amodal:
            amodal_masks = convert_to_mask([obj["segmentation"] for obj in annos])
            visible_masks = convert_to_mask([obj["visible_mask"] for obj in annos])
            occluded_masks = convert_to_mask([obj["occluded_mask"] for obj in annos])
        else:
            visible_masks = convert_to_mask([obj["visible_mask"] for obj in annos])

        if amodal:
            target.gt_masks = merge_bitmask(amodal_masks)
            target.gt_visible_masks = merge_bitmask(visible_masks)
            target.gt_occluded_masks = merge_bitmask(occluded_masks)
            target.gt_occluded_rate = torch.Tensor([obj["occluded_rate"] for obj in annos])
        else:
            target.gt_masks = merge_bitmask(visible_masks)
            target.gt_boxes = target.gt_masks.get_bounding_boxes()

    if not annos:
        return target

    return target
