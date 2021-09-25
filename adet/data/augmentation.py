import random

import numpy as np
from fvcore.transforms import transform, Transform
from detectron2.data.transforms import RandomCrop, StandardAugInput
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.structures import BoxMode

from PIL import Image
import cv2
import pyfastnoisesimd as fns

def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return Transform.CropTransform(0, 0, image_size[1], image_size[0])

    return Transform.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    Modified to support RGB-D image (W, H, 6)
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) == 3 and img.shape[2] == 6:
                depth_image = img[:, :, 3:6]
                pil_image = Image.fromarray(img[:, :, :3])
            elif len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
            elif len(img.shape) == 3 and img.shape[2] == 6:
                depth_image = cv2.resize(depth_image, (self.new_w, self.new_h), cv2.INTER_NEAREST)
                ret = np.concatenate([pil_image, depth_image], -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class Resize(Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )



# Codes from https://github.com/facebookresearch/detectron2/blob/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd/projects/PointRend/point_rend/color_augmentation.py
class ColorAugSSDTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).
    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Implementation based on:
     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp
     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        self.rgbd = False
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if img.shape[-1] == 6:
            self.rgbd = True
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if random.randrange(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            if self.rgbd:
                depth = img[:, :, 3:6]
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            if self.rgbd:
                img = np.concatenate([img, depth], -1)
        return img
    
    def hue(self, img):
        if random.randrange(2):
            if self.rgbd:
                depth = img[:, :, 3:6]
                img = img[:, :, :3]
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            if self.rgbd:
                img = np.concatenate([img, depth], -1)
        return img
    
def perlin_noise(frequency, width, height):

    noise = fns.Noise()
    noise.NoiseType = 2 # perlin noise
    noise.frequency = frequency
    result = noise.genAsGrid(shape=[height, width], start=[0,0])
    return result

def PerlinDistortion(image, width, height):
    """
    """
    # sample distortion parameters from noise vector
    fx = np.random.uniform(0.0001, 0.1)
    fy = np.random.uniform(0.0001, 0.1)
    fz = np.random.uniform(0.01, 0.1)
    wxy = np.random.uniform(0, 10)
    wz = np.random.uniform(0, 0.005)
    cnd_x = wxy * perlin_noise(fx, width, height)
    cnd_y = wxy * perlin_noise(fy, width, height)
    cnd_z = wz * perlin_noise(fz, width, height)

    cnd_h = np.array(list(range(height)))
    cnd_h = np.expand_dims(cnd_h, -1)
    cnd_h = np.repeat(cnd_h, width, -1)
    cnd_w = np.array(list(range(width)))
    cnd_w = np.expand_dims(cnd_w, 0)
    cnd_w = np.repeat(cnd_w, height, 0)

    noise_cnd_h = np.int16(cnd_h + cnd_x)
    noise_cnd_h = np.clip(noise_cnd_h, 0, (height - 1))
    noise_cnd_w = np.int16(cnd_w + cnd_y)
    noise_cnd_w = np.clip(noise_cnd_w, 0, (width - 1))

    new_img = image[(noise_cnd_h, noise_cnd_w)]
    new_img = new_img = new_img + cnd_z
    return new_img.astype(np.float32)