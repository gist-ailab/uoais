# Unseen Object Amodal Instance Segmentation (UOAIS)

Seunghyeok Back, Joosoon Lee, Taewon Kim, Sangjun Noh, Raeyoung Kang, Seongho Bak, Kyoobin Lee 


This repository contains source codes for the paper "Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling." (ICRA 2022)

[[Paper]](https://ieeexplore.ieee.org/abstract/document/9811646) [[ArXiv]](https://arxiv.org/abs/2109.11103) [[Project Website]](https://sites.google.com/view/uoais) [[Video]](https://youtu.be/rDTmXu6BhIU) 

<img src="./imgs/demo.gif" height="300">


## Updates & TODO Lists
- [X] (2021.09.26) UOAIS-Net has been released 
- [X] (2021.11.15) Inference codes for kinect azure and OSD dataset.
- [X] (2021.11.22) ROS nodes for kinect azure and realsense D435
- [X] (2021.12.22) Train and evaluation codes on OSD and OCID dataset + OSD-Amodal annotation


## Getting Started

### Environment Setup

Tested on Titan RTX with python 3.7, pytorch 1.8.0, torchvision 0.9.0, CUDA 10.2 / 11.1 and detectron2 v0.5 / v0.6 (try torch 1.9 if it does not work)

1. Download source codes and checkpoints
```
git clone https://github.com/gist-ailab/uoais.git
cd uoais
mkdir output
```
2. Download checkpoints at [GDrive](https://drive.google.com/drive/folders/1D5hHFDtgd5RnX__55MmpfOAM83qdGYf0?usp=sharing) 

3. Move the `R50_depth_mlc_occatmask_hom_concat` and `R50_rgbdconcat_mlc_occatmask_hom_concat` to the `output` folder.

4. Move the `rgbd_fg.pth` to the `foreground_segmentation` folder


5. Set up a python environment
```
conda create -n uoais python=3.8
conda activate uoais
pip install torch torchvision 
pip install shapely torchfile opencv-python pyfastnoisesimd rapidfuzz termcolor
```
6. Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

7. Build custom [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) inside this repo (at the `uoais` folder)
```
python setup.py build develop 
```

### Run on sample OSD dataset

<img src="./imgs/demo.png" height="200">

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python tools/run_on_OSD.py --use-cgnet --dataset-path ./sample_data --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth) + CG-Net (foreground segmentation)
python tools/run_on_OSD.py --use-cgnet --dataset-path ./sample_data  --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (RGB-D)
python tools/run_on_OSD.py --dataset-path ./sample_data --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python tools/run_on_OSD.py --dataset-path ./sample_data --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```


### Run with ROS

1. Realsense D-435 ([realsense-ros](https://github.com/IntelRealSense/realsense-ros) is required.)
```
# launch realsense2 driver
roslaunch realsense2_camera rs_aligned_depth.launch
# launch uoais node
roslaunch uoais uoais_rs_d435.launch 
# or you can use rosrun
rosrun uoais uoais_node.py _mode:="topic"

```

2. Azure Kinect ([Azure_kinect_ROS_Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver) is required)

```
# launch azure kinect driver
roslaunch azure_kinect_ros_driver driver.launch
# launch uoais node
roslaunch uoais uoais_k4a.launch
```

#### Topics & service
- `/uoais/vis_img` (`sensor_msgs/Image`): visualization results
- `/uoais/results` (`uoais/UOAISResults`): UOAIS-Net predictions (`mode:=topic`)
- `/get_uoais_results` (`uoais/UOAISRequest`): UOAIS-Net predictions (`mode:=service`)

#### Parameters
- `mode` (`string`): running mode of ros node (`topic` or `service`)
- `rgb` (`string`):  topic name of the input rgb
- `depth` (`string`):  topic name of the input depth
- `camera_info` (`string`):  topic name of the input camera info
- `use_cgnet` (`bool`): use CG-Net [1] for foreground segmentation or not 
- `use_planeseg` (`bool`): use RANSAC for plane segmentation or not
- `ransac_threshold` (`float`): max distance a point can be from the plane model



### Run without ROS


1. Realsense D-435 ([librealsense](https://github.com/IntelRealSense/librealsense) and [pyrealsense2](https://pypi.org/project/pyrealsense2/) are required.)

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python tools/rs_demo.py --use-cgnet --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
UOAIS-Net (depth) + CG-Net (foreground segmentation)
python tools/rs_demo.py --use-cgnet --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (RGB-D)
python tools/rs_demo.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python tools/rs_demo.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```


2. Azure Kinect ([Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) and [pyk4a](https://github.com/etiennedub/pyk4a) are required.)

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python tools/k4a_demo.py --use-cgnet --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
UOAIS-Net (depth) + CG-Net (foreground segmentation)
python tools/k4a_demo.py --use-cgnet --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (RGB-D)
python tools/k4a_demo.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python tools/k4a_demo.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```


## Train & Evaluation

### Dataset Preparation

1. Download `UOAIS-Sim.zip` and `OSD-Amodal-annotations.zip` at [GDrive](https://drive.google.com/drive/folders/1D5hHFDtgd5RnX__55MmpfOAM83qdGYf0?usp=sharing) 
2. Download `OSD-0.2-depth.zip` at [OSD](https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/osd/). [2]
4. Download `OCID dataset` at [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/). [3]
5. Extract the downloaded datasets and organize the folders as follows
```
uoais
├── output
└── datasets
       ├── OCID-dataset # for evaluation on indoor scenes
       │     └──ARID10
       │     └──ARID20
       │     └──YCB10
       ├── OSD-0.20-depth # for evaluation on tabletop scenes
       │     └──amodal_annotation # OSD-amodal
       │     └──annotation
       │     └──disparity
       │     └──image_color
       │     └──occlusion_annotation # OSD-amodal
       └── UOAIS-Sim # for training
              └──annotations
              └──train
              └──val
```

### Train on UOAIS-Sim
```
# UOAIS-Net (RGB-D) 
python train_net.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth) 
python train_net.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml 
```


### Evaluation on OSD dataset

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml --use-cgnet
# UOAIS-Net (depth) + CG-Net (foreground segmentation)
python eval/eval_on_OSD.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml --use-cgnet
```
This code evaluates the UOAIS-Net that was trained on a single seed (7), thus the metrics from this code and the paper (an average of seeds 7, 77, 777) can be different.

### Evaluation on OCID dataset

```
# UOAIS-Net (RGB-D)
python eval/eval_on_OCID.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python eval/eval_on_OCID.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```

### Visualization on OSD dataset

```
python tools/run_on_OSD.py --use-cgnet --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
```


## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.

## Notes

The codes of this repository are built upon the following open sources. Thanks to the authors for sharing the code!
- Instance segmentation based on [Detectron2](https://github.com/facebookresearch/detectron2) and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
- Evaluation codes are modified from [4] [UCN](https://github.com/NVlabs/UnseenObjectClustering) and [5] [VRSP-Net](https://github.com/YutingXiao/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior). 
- Synthetic dataset is generated with [6] [BlenderProc](https://github.com/DLR-RM/BlenderProc)


## Citation
If you use our work in a research project, please cite our work:
```
@inproceedings{back2022unseen,
  title={Unseen object amodal instance segmentation via hierarchical occlusion modeling},
  author={Back, Seunghyeok and Lee, Joosoon and Kim, Taewon and Noh, Sangjun and Kang, Raeyoung and Bak, Seongho and Lee, Kyoobin},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)},
  pages={5085--5092},
  year={2022},
  organization={IEEE}
}
```

## References
```
[1] SUN, Yao, et al. Cg-net: Conditional gis-aware network for individual building segmentation in vhr sar images. IEEE Transactions on Geoscience and Remote Sensing, 2021, 60: 1-15.
[2] Richtsfeld, Andreas, et al. "Segmentation of unknown objects in indoor environments." 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012.
[3] Suchi, Markus, et al. "EasyLabel: a semi-automatic pixel-wise object annotation tool for creating robotic RGB-D datasets." 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019.
[4] Xiang, Yu, et al. "Learning rgb-d feature embeddings for unseen object instance segmentation." Conference on Robot Learning (CoRL). 2020.
[5] Xiao, Yuting, et al. "Amodal Segmentation Based on Visible Region Segmentation and Shape Prior." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 4. 2021.
[6] DENNINGER, Maximilian, et al. Blenderproc. arXiv preprint arXiv:1911.01911, 2019.
```
