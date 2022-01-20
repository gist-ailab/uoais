import argparse
import os
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
import pyrealsense2 as rs

from utils import *
from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor
from foreground_segmentation.model import Context_Guided_Network

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
        default=0.7,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter our background instances or not"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to forground segmentation weight"
    )
    return parser


if __name__ == "__main__":

    # UOAIS-Net
    args = get_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # CG-Net (foreground segmentation)
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # RealSense

    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)


    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # aligned_depth_frame = decimation.process(aligned_depth_frame)
        aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
        aligned_depth_frame = spatial.process(aligned_depth_frame)
        aligned_depth_frame = temporal.process(aligned_depth_frame)
        aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
        # aligned_depth_frame = hole_filling.process(aligned_depth_frame)
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        depth = np.asanyarray(aligned_depth_frame.get_data()) * depth_scale * 1000
        rgb_img = np.asanyarray(color_frame.get_data()) 

        depth_img = normalize_depth(depth)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        # CG-Net inference
        if args.use_cgnet:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

        preds = instances.pred_masks.detach().cpu().numpy() 
        pred_visibles = instances.pred_visible_masks.detach().cpu().numpy() 
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
        pred_occs = instances.pred_occlusions.detach().cpu().numpy() 
        
        # filter out the background instances
        if args.use_cgnet:
            remove_idxs = []
            for i, pred_visible in enumerate(pred_visibles):
                iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
                if iou < 0.5: 
                    remove_idxs.append(i)
            preds = np.delete(preds, remove_idxs, 0)
            pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
            bboxes = np.delete(bboxes, remove_idxs, 0)
            pred_occs = np.delete(pred_occs, remove_idxs, 0)
        
        # reorder predictions for visualization
        idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
        preds, pred_occs, bboxes = preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf]
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)

        if args.use_cgnet:
            vis_fg = np.zeros_like(rgb_img) 
            vis_fg[:, :, 1] = fg_output*255
            vis_img = cv2.addWeighted(vis_img, 0.8, vis_fg, 0.2, 0)

        vis_all_img = np.hstack([rgb_img, depth_img, vis_img])

        cv2.imshow("ESC: quit" + args.config_file, vis_all_img)
        k = cv2.waitKey(1)
        if k == 27: # esc
            cv2.destroyAllWindows()
            break  



    
