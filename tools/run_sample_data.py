import os
import cv2
import argparse
import glob
import numpy as np

import torch

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.utils.visualizer import Visualizer, visualize_pred_amoda_occ
from adet.config import get_cfg
from adet.utils.post_process import detector_postprocess, DefaultPredictor
import time
import glob
import cv2


import argparse
import multiprocessing as mp
import os
import cv2
from adet.utils.visualizer import Visualizer, visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor
import numpy as np
from adet.config import get_cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--image-folder",
        default="sample_data",
        metavar="FILE",
        help="path to sample data folder",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)

    # naive version
    rgb_imgs = sorted(glob.glob(args.image_folder + "/rgb_*.png"))
    depth_imgs = sorted(glob.glob(args.image_folder + "/depth_*.png"))

    for idx, (rgb_img, depth_img) in enumerate(zip(rgb_imgs, depth_imgs)):

        rgb_img = cv2.imread(rgb_img)
        depth_img = cv2.imread(depth_img)
        if cfg.INPUT.DEPTH_ONLY:
            input_img = np.float32(depth_img)
        else:
            input_img = np.concatenate([rgb_img, np.float32(depth_img)/255], -1)
        
        start_time = time.time()
        outputs = predictor(input_img[:, :, :])
        instances = detector_postprocess(outputs['instances'], 480, 640).to('cpu') 
        print("Inference took {} seconds for {}-th image".format(round(time.time() - start_time, 3), idx))

        # reorder predictions for visualization
        preds = instances.pred_masks.detach().cpu().numpy() 
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
        pred_occs = instances.pred_occlusions.detach().cpu().numpy() 
        idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
        preds, pred_occs, bboxes = preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf]
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)
        vis_all_img = np.hstack([rgb_img, depth_img, vis_img])

        cv2.imshow("sample_data_{}".format(idx), vis_all_img)
        k = cv2.waitKey(0)
        if k == 27: # esc
            break  
        cv2.destroyAllWindows()
