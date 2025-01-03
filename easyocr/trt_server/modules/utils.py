import glob
import random
import argparse
import os.path as osp
from typing import Callable

import cv2
import numpy as np
from PIL import Image


class Flag_config:
    """stores configurations for prediction"""

    def __init__(self):
        pass


class DataStreamer(object):
    """Iterable DataStreamer class for generating numpy arr images
    Generates orig image and pre-processed image

    Attributes
    ----------
    src_path : str
        path to a single image/video or path to directory containing images
    media_type : str
        inference media_type "image" or "video"
    preprocess_func : Callable function
        preprocessesing function applied to PIL images
    """

    def __init__(self, src_path: str, media_type: str = "image", preprocess_func: Callable = None):
        if media_type not in {'video', 'image'}:
            raise NotImplementedError(
                f"{media_type} not supported in streamer. Use video or image")
        self.img_path_list = []
        self.vid_path_list = []
        self.idx = 0
        self.media_type = media_type
        self.preprocess_func = preprocess_func

        if media_type == "video":
            if osp.isfile(src_path):
                self.vid_path_list.append(src_path)
                self.vcap = cv2.VideoCapture(src_path)
            elif osp.isdir(src_path):
                raise NotImplementedError(
                    f"dir iteration supported for video media_type. {src_path} must be a video file")
        elif media_type == "image":
            if osp.isfile(src_path):
                self.img_path_list.append(src_path)
            elif osp.isdir(src_path):
                img_exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg']
                for ext in img_exts:
                    self.img_path_list.extend(
                        glob.glob(osp.join(src_path, ext)))

    def __iter__(self):
        return self

    def __next__(self):
        orig_img = None
        if self.media_type == 'image':
            if self.idx < len(self.img_path_list):
                orig_img = Image.open(self.img_path_list[self.idx])
                self.idx += 1
        elif self.media_type == 'video':
            if self.idx < len(self.vid_path_list):
                ret, frame = self.vcap.read()
                if ret:
                    orig_img = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    self.idx += 1
        if orig_img is not None:
            proc_img = None
            if self.preprocess_func is not None:
                proc_img = self.preprocess_func(orig_img)
                proc_img = np.expand_dims(proc_img, axis=0)
            return np.array(orig_img), proc_img
        raise StopIteration


def parse_arguments(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_path',
                        required=True,  type=str,
                        help='Path to Input: Video File or Image file')
    parser.add_argument('-m', '--media_type',
                        default='image', type=str,
                        choices=('image', 'video'),
                        help='Type of Input: image, video. Default: image')
    parser.add_argument('-o', '--output_dir',
                        default='output',  type=str,
                        help='Output directory. Default: output')
    parser.add_argument('-t', '--detection_threshold',
                        default=0.6,  type=float,
                        help='Detection Threshold. Default: 0.6')
    parser.add_argument('-g', '--grpc_port',
                        default="8994",
                        help='grpc port where triton-server is exposed. Default: 8994')
    parser.add_argument('--debug',
                        default=True,
                        help='Debug Mode Default: True')

    return parser.parse_args()


def plot_one_box(bbox, img, wscale=1, hscale=1, color=None, label=None, line_thickness=None) -> None:
    """
    Plot one bounding box on image img
    args
        bboxes: bounding boxes in xyxy format (x1,y1,x2,y2)
        img: image in (H,W,C) numpy.ndarray fmt
        wscale: multiplication factor for width (default 1 if no scaling required)
        hscale: multiplication factor for height (default 1 if no scaling required)
    """
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1 = (int(bbox[0] * wscale), int(bbox[1] * hscale))
    c2 = (int(bbox[2] * wscale), int(bbox[3] * hscale))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def resize_maintaining_aspect(img, width, height):
    """
    If width and height are both None, no resize is done
    If either width or height is None, resize maintaining aspect
    """
    old_h, old_w, _ = img.shape

    if width is not None and height is not None:
        new_w, new_h = width, height
    elif width is None and height is not None:
        new_h = height
        new_w = (old_w * new_h) // old_h
    elif width is not None and height is None:
        new_w = width
        new_h = (new_w * old_h) // old_w
    else:
        # no resizing done if both width and height are None
        return img
    img = cv2.resize(img, (new_w, new_h))
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0,
                     mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0,
                     variance[2] * 255.0], dtype=np.float32)
    return img
