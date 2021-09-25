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
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    print(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)

    # naive version
    cap = cv2.VideoCapture("/home/seung/Workspace/datasets/UOAIS/CATER/CATER_new_005781.avi")
    success, img = cap.read()
    fno = 0
    while success:
        # read next frame
        success, img = cap.read()
            # use PIL, to be consistent with evaluation

        img = cv2.resize(img, cfg.IMG_SIZE)

        outputs = predictor(img[:, :, ::-1])

        vis_img = visualize_pred_amoda_occ(color, preds, bboxs, pred_occ)

        cv2.imshow(args.config_file.split('/')[-1], pred_vis)
        k = cv2.waitKey(1)
        if k == 27:
            break  # esc to quit



    