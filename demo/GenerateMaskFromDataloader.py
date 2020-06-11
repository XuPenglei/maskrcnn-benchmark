from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2
from skimage import transform
import skimage

import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils import cv2_util

from maskrcnn_benchmark.config import cfg


def overlay_boxes(image, predictions):
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    color_single = [255, 0, 0]

    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color_single), 1
        )
    return image


def overlay_mask(image, predictions):
    masks = predictions.get_field("mask").numpy().squeeze()
    size = image.shape[:2]
    labels = predictions.get_field("labels")
    color_single = [255, 0, 0]
    for mask in masks:
        mask = transform.resize(mask, size, preserve_range=True, order=3).astype(np.uint8)
        thresh = mask[:, :, None].astype(np.uint8)
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color_single, 1)

    composite = image

    return composite


def build(cfg, is_train=False):
    data_loader_val = make_data_loader(cfg, is_train=is_train, is_distributed=False, start_iter=10000)
    model = build_detection_model(cfg)
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    save_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    if is_train:
        return data_loader_val, model
    return data_loader_val[0], model


def predict_from_dataloader(cfg):
    data_loader, model = build(cfg, True)
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device('cpu')
    pred_list = []
    masker = Masker(threshold=0.5, padding=1)
    ids = []
    dataset = data_loader.dataset
    for images, targets, img_ids in data_loader:
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            predictions = model(images, targets)
        predictions = [o.to(cpu_device) for o in predictions]
        pred_list.extend(predictions)
        ids.extend(img_ids)
    pred_transformed_list = {}
    for prediction, id in zip(pred_list, ids):
        # prediction = singleIMG_pred.resize((width, height))
        if prediction.has_field("mask"):
            masks = prediction.get_field("mask")
            masks = masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        pred_transformed_list.update({id: prediction})
    return pred_transformed_list, dataset


def overlay_pred(img, prediction, mask_on=True):
    result = img.copy()
    result = overlay_boxes(result, prediction)
    if mask_on:
        result = self.overlay_mask(result, top_predictions)
    return result


def main():
    config_file = "../configs/vertex_only_R_50_FPN_1x.yaml"
    img_dir = r"E:\ResearchDOC\term2\MSRCNN_polyrnn\maskrcnn-benchmark\dataForLittleTest\IMG_vertexonly"
    cfg.merge_from_file(config_file)
    preds, dataset = predict_from_dataloader(cfg)
    for image_id, p in preds.items():
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        img_w = img_info["width"]
        img_h = img_info["height"]
        path = os.path.join(img_dir, img_info["file_name"])
        img = np.array(Image.open(path).convert("RGB"))
        p = p.resize((img_w, img_h))
        result = overlay_mask(img, p)
        plt.imshow(result)
        plt.show()

if __name__ == "__main__":
    main()
