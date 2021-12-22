# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

def _get_uoais_instances_meta():

    # !TODO: modify this for uoais dataset
    thing_ids = [1]
    thing_colors = [[92, 85, 25]]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = ['object']
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_wisdom_instances_meta():

    # !TODO: modify this for uoais dataset
    thing_ids = [1]
    thing_colors = [[92, 85, 25]]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = ['object']
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "uoais":
        return _get_uoais_instances_meta()
    elif dataset_name == "wisdom":
        return _get_wisdom_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
