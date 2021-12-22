# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


from .uoais import load_uoais_json

"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

__all__ = ["register_uoais_instances"]



def register_uoais_instances(name, metadata, json_file, image_root, amodal):
    """
    Register a dataset in uoais's json annotation format for
    instance detection
    Args:
        name (str): the name that identifies a dataset, e.g. "d2sa_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_uoais_json(json_file, image_root, name,))
    evaluator_type = "amodal" if amodal else "coco"

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type=evaluator_type, **metadata
    )
