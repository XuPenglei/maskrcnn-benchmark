from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2

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
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    color_single = [255, 0, 0]
    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None].astype(np.uint8)
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color_single, 1)
        # image = cv2.addWeighted(image,0.8,mask[0][:,:,None].repeat(3,-1),0.2,0)

    composite = image

    return composite


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


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


def predict_from_dataloader(cfg, original_size):
    data_loader, model = build(cfg, True)
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device('cpu')
    pred_list = []
    masker = Masker(threshold=0.5, padding=1)
    img_list = []
    for images, targets, _ in data_loader:
        img_list.append(images)
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            predictions = model(images, targets)
        predictions = [o.to(cpu_device) for o in predictions]
        pred_list.append(predictions)
    height, width = original_size
    pred_transformed_list = []
    for prediction in pred_list:
        prediction = prediction.resize((width, height))
        if prediction.has_field("mask"):
            masks = prediction.get_field("mask")
            masks = masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        pred_transformed_list.append(prediction)
    return pred_transformed_list, img_list


def overlay_pred(img, prediction, mask_on=True):
    result = img.copy()
    result = overlay_boxes(result, prediction)
    if mask_on:
        result = self.overlay_mask(result, top_predictions)
    return result


def main():
    config_file = "../configs/vertex_only_R_50_FPN_1x.yaml"
    cfg.merge_from_file(config_file)
    preds, imgs = predict_from_dataloader(cfg, [800, 800])


if __name__ == "__main__":
    main()
