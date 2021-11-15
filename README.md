# Unseen Object Amodal Instance Segmentation (UOAIS)

Seunghyeok Back, Joosoon Lee, Taewon Kim, Sangjun Noh, Raeyoung Kang, Seongho Bak, Kyoobin Lee 


This repository contains source codes for the paper "Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling."

[[Paper]](https://arxiv.org/abs/2109.11103) [[Project Website]](https://sites.google.com/view/uoais) [[Video]](https://youtu.be/rDTmXu6BhIU) 

<img src="./imgs/demo.gif" height="200">


## Updates & TODO Lists
- [X] (2021.09.26) UOAIS-Net has been released 
- [X] (2021.11.15) inference codes for kinect azure and OSD dataset.
- [ ] Add train and evaluation code
- [ ] Release synthetic dataset (UOAIS-Sim) and amodal annotation (OSD-Amodal)
- [ ] Add ROS inference node (kinect azure, realsense)


## Getting Started

### Environment Setup

Tested on Titan RTX with python 3.7, pytorch 1.8.0, torchvision 0.9.0, CUDA 10.2.

1. Download
```
git clone https://github.com/gist-ailab/uoais.git
cd uoais
mkdir output
```
Download the checkpoint at [GDrive](https://drive.google.com/drive/folders/1D5hHFDtgd5RnX__55MmpfOAM83qdGYf0?usp=sharing) 

Move the `R50_depth_mlc_occatmask_hom_concat` and `R50_rgbdconcat_mlc_occatmask_hom_concat` to the `output` folder.

Move the `rgbd_fg.pth` to the `foreground_segmentation` folder


2. Set up a python environment
```
conda create -n uoais python=3.7
conda activate uoais
pip install torch torchvision 
pip install shapely torchfile opencv-python pyfastnoisesimd rapidfuzz
```
3. Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only)
4. Build and install custom [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
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
python tools/run_on_OSD.py --use-fg --dataset-path {OSD dataset path} --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
```


### Run with Kinect Azure

[Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) and [pyk4a](https://github.com/etiennedub/pyk4a) are required.

```
# UOAIS-Net (RGB-D) + Foreground Segmentation
python tools/k4a_demo.py --use-fg --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
UOAIS-Net (depth) + Foreground Segmentation 
python tools/k4a_demo.py --use-fg --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (RGB-D)
python tools/k4a_demo.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python tools/k4a_demo.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```

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
