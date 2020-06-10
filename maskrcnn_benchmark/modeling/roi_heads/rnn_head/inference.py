import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList
from ..mask_head.inference import Masker

import cv2
import matplotlib.pyplot as plt

def class_to_grid(poly,grid_size,mask=None):
    grid_polygon = []
    for i in poly:
        if i < grid_size * grid_size:
            x = int(i % grid_size)
            y = int(i / grid_size)
            grid_polygon.append([x, y])
        else:
            break
    if mask is not None:
        assert mask.shape[0]==mask.shape[1]==grid_size
        cv2.fillPoly(mask,[np.array(grid_polygon)],[1])
        return mask
    return grid_polygon

def vertexs_to_mask(vertexs,boxes,grid_size,device):
    boxes_per_img = [len(b) for b in boxes]
    vertexs = vertexs.split(boxes_per_img,dim=0)
    masks = []
    for vers_perIMG in vertexs:
        masks_perIMG = []
        vers_perIMG = vers_perIMG.cpu().numpy()
        for vs in vers_perIMG:
            mask = np.zeros((grid_size,grid_size),dtype=np.uint8)
            # vs_class = np.argmax(vs,axis=1)
            mask = class_to_grid(vs,grid_size,mask)
            masks_perIMG.append(mask)
        masks_perIMG = np.expand_dims(np.array(masks_perIMG),axis=1)
        masks.append(torch.from_numpy(masks_perIMG).to(device))
    return masks

class VertexPostProcessor(nn.Module):
    def __init__(self, grid_size,masker=None ):
        super(VertexPostProcessor, self).__init__()
        self.masker = masker
        self.grid_size = grid_size

    def forward(self, x, boxes):
        masks = vertexs_to_mask(x,boxes,grid_size=self.grid_size,device=x.device)
        if self.masker:
            masks = self.masker(masks)

        results = []
        for mask, box in zip(masks,boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", mask)
            results.append(bbox)

        return results


def make_roi_vertex_post_processor(cfg):
    if cfg.MODEL.ROI_RNN_HEAD.POSTPROCESS_MASKS:
        mask_threshold = 0.5
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    vertex_post_processor = VertexPostProcessor(grid_size=cfg.MODEL.ROI_RNN_HEAD.POOLER_RESOLUTION,
                                                masker=masker)
    return vertex_post_processor





