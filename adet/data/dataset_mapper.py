import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .detection_utils import (annotations_to_instances, transform_instance_annotations)
from .augmentation import ColorAugSSDTransform, Resize, PerlinDistortion

import cv2
import imageio
import torch.nn.functional as F
import random

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.img_size = cfg.INPUT.IMG_SIZE  # width, height
        self.amodal = cfg.INPUT.AMODAL
        self.depth = cfg.INPUT.DEPTH
        self.rgbd_fusion = cfg.MODEL.RGBD_FUSION
        self.depth_min, self.depth_max = cfg.INPUT.DEPTH_RANGE
        self.recompute_boxes = cfg.INPUT.RECOMPUTE_BOXES
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.color_aug = cfg.INPUT.COLOR_AUGMENTATION
        self.depth_only = cfg.INPUT.DEPTH_ONLY
        self.perlin_distortion = cfg.INPUT.PERLIN_DISTORTION

        if is_train:
            self.is_wisdom = "wisdom" in cfg.DATASETS.TRAIN[0] 
        else:
            self.is_wisdom = "wisdom" in cfg.DATASETS.TEST[0]

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False
        cr = cfg.INPUT.CROP_RATIO

        if self.color_aug and is_train:
            if self.depth_only:
               self.augmentation_lists = [
                T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
                T.RandomFlip(0.5),
                Resize((self.img_size[1], self.img_size[0]))
                ]
            else:
                self.augmentation_lists = [
                    T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
                    ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT),
                    T.RandomFlip(0.5),
                    Resize((self.img_size[1], self.img_size[0]))
                    ]
        elif not self.color_aug and is_train:
            self.augmentation_lists = [
                T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
                T.RandomFlip(0.5),
                Resize((self.img_size[1], self.img_size[0]))
            ]
            
        else:
            self.augmentation_lists = [
                Resize((self.img_size[1], self.img_size[0]))
            ]      
        logging.getLogger(__name__).info(
            "Augmentation used in training: {}".format(self.augmentation_lists)
            )
        self.augmentation_lists = T.AugmentationList(self.augmentation_lists)


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )

        if self.depth:
            if self.is_wisdom: 
                depth = cv2.imread(dataset_dict["depth_file_name"])
            else: 
                depth = imageio.imread(dataset_dict["depth_file_name"]).astype(np.float32)
                if self.perlin_distortion and self.is_train:
                    # if random.random() > 0.5:
                    depth = PerlinDistortion(depth, *self.img_size)
                    # depth = PerlinDistortion(depth, depth.shape[1], depth.shape[0])
                depth[depth > self.depth_max] = self.depth_max
                depth[depth < self.depth_min] = self.depth_min
                depth = (depth - self.depth_min) / (self.depth_max - self.depth_min) * 255
                depth = np.expand_dims(depth, -1)
                depth = np.uint8(np.repeat(depth, 3, -1))
            # depth = cv2.resize(depth, tuple(self.img_size[::-1]), interpolation=cv2.INTER_NEAREST)
        if self.depth and self.depth_only:
            image = depth
        if self.depth and self.rgbd_fusion == "late":
            image = np.concatenate([image, depth], -1)
        elif self.depth and self.rgbd_fusion == "early":
            pass

        boxes = np.asarray([BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                    )
                        for instance in dataset_dict["annotations"]])

        # apply the color augmentation
        aug_input = T.AugInput(image, boxes=boxes)
        transforms = self.augmentation_lists(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # if self.depth and self.rgbd_fusion != "none":
        #     depth = torch.as_tensor(
        #         np.ascontiguousarray(depth.transpose(2, 0, 1))
        #     )
        #     if self.rgbd_fusion == "early":
        #         dataset_dict["image"] = torch.cat([dataset_dict["image"], depth[0, :, :].unsqueeze(0)], 0)
        #     elif self.rgbd_fusion == "late":
        #         dataset_dict["image"] = torch.cat([dataset_dict["image"], depth], 0)
        #     else:
        #         raise NotImplementedError
        # elif self.depth and self.rgbd_fusion == "none":
        #     depth = torch.as_tensor(
        #         np.ascontiguousarray(depth.transpose(2, 0, 1))
        #     )
        #     dataset_dict["image"] = depth[0, :, :].unsqueeze(0)


        # if not self.is_train:
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     dataset_dict.pop("pano_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            # USER: Implement additional transformations if you have other types of data
            # transform_instance_annotations converts rle to bitmask
            annos = [
                transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            dataset_dict["annotations"] = annos
            instances = annotations_to_instances(
                annos, image_shape, mask_format="rle", amodal=self.amodal
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
