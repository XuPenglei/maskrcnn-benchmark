# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3



@registry.ROI_VERTEX_FEATURE_EXTRACTORS.register("RNNFPNFeatureExtractor")
class RNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for vertexes predict
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(RNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RNN_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_RNN_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_RNN_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_RNN_HEAD.USE_GN
        layers = cfg.MODEL.ROI_RNN_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_RNN_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "rnn_fe{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_rnn_feature_extractor(cfg, in_channels):
    func = registry.ROI_VERTEX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RNN_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
