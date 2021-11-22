# Unseen Object Amodal Instance Segmentation (UOAIS)

Seunghyeok Back, Joosoon Lee, Taewon Kim, Sangjun Noh, Raeyoung Kang, Seongho Bak, Kyoobin Lee 


This repository contains source codes for the paper "Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling."

[[Paper]](https://arxiv.org/abs/2109.11103) [[Project Website]](https://sites.google.com/view/uoais) [[Video]](https://youtu.be/rDTmXu6BhIU) 

<img src="./imgs/demo.gif" height="200">


## Updates & TODO Lists
- [X] (2021.09.26) UOAIS-Net has been released 
- [X] (2021.11.15) inference codes for kinect azure and OSD dataset.
- [X] (2021.11.22) ROS nodes for kinect azure and realsense D435
- [ ] Add train and evaluation code
- [ ] Release synthetic dataset (UOAIS-Sim) and amodal annotation (OSD-Amodal)


## Getting Started

### Environment Setup

Tested on Titan RTX with python 3.7, pytorch 1.8.0, torchvision 0.9.0, CUDA 10.2 / 11.1 and detectron2 v0.5 / v0.6

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
conda create -n uoais python=3.7
conda activate uoais
pip install torch torchvision 
pip install shapely torchfile opencv-python pyfastnoisesimd rapidfuzz
```
6. Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only)

7. Build and install custom [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
```
python setup.py build develop 
```

### Run on sample OSD dataset

<img src="./imgs/demo.png" height="200">

```
# UOAIS-Net (RGB-D) + Foreground Segmentation
python tools/run_on_OSD.py --use-fg --dataset-path ./sample_data --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth) + Foreground Segmentation
python tools/run_on_OSD.py --use-fg --dataset-path ./sample_data  --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (RGB-D)
python tools/run_on_OSD.py --dataset-path ./sample_data --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python tools/run_on_OSD.py --dataset-path ./sample_data --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```


### Run on full OSD dataset

Download `OSD-0.2-depth.zip` at [OSD](https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/osd/) and extract it.
```
python tools/run_on_OSD.py --use-cgnet --dataset-path {OSD dataset path} --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
```


### Run with Kinect Azure

[Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) and [pyk4a](https://github.com/etiennedub/pyk4a) are required.

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

### ROS nodes

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
- `use_cgnet` (`bool`): use CG-Net for foreground segmentation or not
- `use_planeseg` (`bool`): use RANSAC for plane segmentation or not
- `ransac_threshold` (`float`): max distance a point can be from the plane model


## License

This repository is released under the MIT license.


## Citation
If you use our work in a research project, please cite our work:
```
@misc{back2021unseen,
      title={Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling}, 
      author={Seunghyeok Back and Joosoon Lee and Taewon Kim and Sangjun Noh and Raeyoung Kang and Seongho Bak and Kyoobin Lee},
      year={2021},
      eprint={2109.11103},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
