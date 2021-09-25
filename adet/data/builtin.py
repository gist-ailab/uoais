# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .builtin_meta import _get_builtin_metadata
from .register_uas import register_uas_instances
from .register_wisdom import register_wisdom_instances



_PREDEFINED_SPLITS_UAS = {
    
    "uas_sim_train_amodal": ("UAS-SIM/train", "UAS-SIM/annotations/coco_anns_uas_amodal_train.json"),
    "uas_sim_val_amodal": ("UAS-SIM/val", "UAS-SIM/annotations/coco_anns_uas_amodal_val.json"),
}


def register_all_uas(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UAS.items():
        # Assume pre-defined datasets live in `./datasets`.
        amodal = "amodal" in key
        if "occ" in key:
            md = "uas_occ"
        else:
            md = "uas"
        register_uas_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            amodal=amodal
        )

_PREDEFINED_SPLITS_WISDOM = {
    "wisdom_real_train": ("wisdom/wisdom-real/high-res", "wisdom/wisdom-real/high-res/annotations/coco_anns_wisdom_train.json"),
    "wisdom_real_test": ("wisdom/wisdom-real/high-res", "wisdom/wisdom-real/high-res/annotations/coco_anns_wisdom_test.json"),
}


def register_all_wisdom(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_WISDOM.items():
        # Assume pre-defined datasets live in `./datasets`.
        if "occ" in key:
            md = "wisdom_occ"
        else:
            md = "wisdom"
        register_wisdom_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



# Register them all under "./datasets"
register_all_uas()
register_all_wisdom()
