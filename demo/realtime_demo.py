# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import pyk4a
from pyk4a import Config, PyK4A
from detectron2.data.detection_utils import read_image
from adet.utils.visualizer import Visualizer
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from adet.config import get_cfg
import torch
# from centermask.config import get_cfg
import numpy as np
# constants
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.utils.visualizer import Visualizer
from adet.config import get_cfg
from detectron2.structures import Instances
from detectron2.layers import paste_masks_in_image



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def draw_predictions(img, metadata, target, resolution=(512, 384)):
    
    try:
        vis = Visualizer(img, metadata=metadata)
        vis = vis.draw_instance_predictions(instances, target)
        vis = cv2.resize(vis.get_image(), resolution)
    except:
        vis = img
    vis = cv2.putText(vis, target, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return vis

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser('UOAIS', add_help=False)
    parser.add_argument("--config-file", 
        default="./configs/ORCNN/R50_1x_lfconv_mlc_ed_np.yaml", 
        metavar="FILE", help="path to config file")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    args = parser.parse_args()    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.5
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                synchronized_images_only=True,
                camera_fps=pyk4a.FPS.FPS_5,
            ))
    k4a.start()
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # naive version
    while True:
        capture = k4a.get_capture()
        color = capture.color[:, :, :3]
        color = cv2.resize(color, (512, 384))
        if cfg.INPUT.DEPTH:
            depth = capture.transformed_depth
            depth[depth < 250] = 250
            depth[depth > 1500] = 1500
            depth = (depth - 250) / (1250) * 255
            depth = np.expand_dims(depth, -1)
            depth = np.uint8(np.repeat(depth, 3, -1))
            depth = cv2.resize(depth, (512, 384))
            mask = 1 * (np.sum(depth, axis=2) == 0)
            inpainted_data = cv2.inpaint(
                depth, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA
            )
            depth = np.where(depth==0, inpainted_data, depth)
            
            img = np.concatenate([color, depth], -1)        
        else:
            img = color

        outputs = predictor(img)
        instances = detector_postprocess(outputs['instances'], 384, 512).to('cpu')

        
        segm_vis = draw_predictions(color, metadata, "pred_masks")
        visible_vis = draw_predictions(color, metadata, "pred_visible_masks")
        occlusion_vis = draw_predictions(color, metadata, "pred_occlusion_masks")
        allinone_visb = np.vstack([np.hstack([color, depth, segm_vis, visible_vis, occlusion_vis])])

        if cfg.MODEL.EDGE_DETECTION:
            edge_vis = draw_predictions(color, metadata, "pred_vis_edges")
            contact_edge_vis = draw_predictions(color, metadata, "pred_contact_edges")
            occluded_edge_vis = draw_predictions(color, metadata, "pred_occluded_edges")
            allinone_visb = np.vstack([np.hstack([color,segm_vis, visible_vis, occlusion_vis]), \
                                        np.hstack([depth, edge_vis, contact_edge_vis, occluded_edge_vis])])
        
        allinone_visb = cv2.resize(allinone_visb, (512*6, 384*3))
        cv2.imshow(args.config_file.split('/')[-1], allinone_visb)
        k = cv2.waitKey(1)
        if k == 27:
            break  # esc to quit



    
