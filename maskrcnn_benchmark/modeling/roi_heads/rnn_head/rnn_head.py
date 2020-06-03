import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_rnn_feature_extractors import make_roi_rnn_feature_extractor
from.RNN_Evaluation import losses,metrics

from .first_v import FirstVertex
from .conv_lstm import AttConvLSTM

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIRnnHead(torch.nn.Module):
    def __init__(self,cfg, in_channels):
        super(ROIRnnHead,self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_rnn_feature_extractor(cfg, in_channels)

