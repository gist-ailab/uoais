import os
import cv2
import argparse
import glob
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from uoamask.config import get_cfg


def main(args):
    # Get image
    image_paths = glob.glob(args.image_folder)
    # Get the configuration ready
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    predictor = DefaultPredictor(cfg)

    for i, image_path in enumerate(tqdm(image_paths)):
        
        im = cv2.imread(image_path)
        im = cv2.resize(im, (512, 384))


        outputs = predictor(im)

        v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        img = v.get_image()[:, :, ::-1]

        if i == 0 and not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "vis_results")):
            os.mkdir(os.path.join(cfg.OUTPUT_DIR, "vis_results"))
        cv2.imwrite('{}/vis_results/output_{}.png'.format(cfg.OUTPUT_DIR, i), img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser('UOIS CenterMask', add_help=False)
    parser.add_argument("--config-file", 
        default="./configs/debug.yaml", 
        metavar="FILE", help="path to config file")
    parser.add_argument("--image-folder", 
        default="./datasets/wisdom/wisdom-real/high-res/color_ims/*.png", 
        metavar="FILE", help="path to config file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    setup_logger()

    main(args)