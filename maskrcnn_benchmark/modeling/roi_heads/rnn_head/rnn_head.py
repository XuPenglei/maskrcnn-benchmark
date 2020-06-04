import torch
from torch import nn
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_rnn_feature_extractors import make_roi_rnn_feature_extractor
from .RNN_Evaluation import losses,metrics
# from .rnn_utils import Target_Preprocessor,xy_to_class,class_to_grid
from .rnn_utils import *
from .first_v import FirstVertex
from .conv_lstm import AttConvLSTM
from .loss import *

def keep_only_positive_boxes(boxes,keep_num=None):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.
    If keep_num is not None, then random select keep_num boxes

    Arguments:
        boxes (list of BoxList)
        keep_num int
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    # positive_boxes = []
    # positive_inds = []
    sampled_boxes = []
    # sampled_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        # positive_boxes.append(boxes_per_image[inds])
        # positive_inds.append(inds_mask)
        if keep_num and len(inds)>keep_num:
            sample_ind = np.random.choice(inds.cpu().numpy().astype(np.int),keep_num,False)
            sampled_boxes.append(boxes_per_image[sample_ind])
            # sampled_inds.append(sample_ind)
    # return positive_boxes, positive_inds, sampled_boxes, sampled_inds
    return sampled_boxes

class ROIRnnHead(torch.nn.Module):
    def __init__(self,cfg, in_channels):
        super(ROIRnnHead,self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_rnn_feature_extractor(cfg, in_channels)
        self.firstV = FirstVertex(cfg)
        self.conv_lstm = AttConvLSTM(cfg)
        self.target_preprocessor = Target_Preprocessor(cfg)
        self.post_processor = ""

    def forward(self,features, proposals, targets=None):

        if self.training:
            fp_beam_size = self.cfg.MODEL.ROI_RNN_HEAD.TRAIN_FP_BEAM_SIZE
            lstm_beam_size = self.cfg.MODEL.ROI_RNN_HEAD.TRAIN_BEAM_SIZE
            temperature = self.cfg.MODEL.ROI_RNN_HEAD.TRAIN_TEMP
            all_proposals = proposals
            proposals = keep_only_positive_boxes(proposals,
                                                 self.cfg.MODEL.ROI_RNN_HEAD.SAMPLE_NUM)
            ver_masks, edge_masks, arr_polys, poly_masks, proposals = \
                self.target_preprocessor.prepare_targets(proposals,targets,self.cfg.MODEL.ROI_RNN_HEAD.BOX_ENLARGE_RATIO)
        else:
            fp_beam_size = self.cfg.MODEL.ROI_RNN_HEAD.TEST_FP_BEAM_SIZE
            lstm_beam_size = self.cfg.MODEL.ROI_RNN_HEAD.TEST_BEAM_SIZE
            temperature = self.cfg.MODEL.ROI_RNN_HEAD.TEST_TEMP
        x = self.feature_extractor(features,proposals)

        edge_logits, vertex_logits, first_logprob,first_v = self.firstV(x,temperature,fp_beam_size)

        poly_class = None
        if targets is not None:
            poly_class = xy_to_class(arr_polys,
                                     grid_size=self.cfg.MODEL.ROI_RNN_HEAD.POOLER_RESOLUTION)
            if self.training:
                first_v = poly_class[:, 0]
                first_logprob = None
            else:
                first_v = poly_class[:, 0]
                first_logprob = None
                lstm_beam_size = 1

        out_dict = self.conv_lstm(
            x,
            first_v,
            poly_class,
            temperature = temperature,
            mode = "train_ce",
            fp_beam_size=fp_beam_size,
            beam_size=lstm_beam_size,
            first_log_prob=first_logprob,
            return_attention=self.cfg.MODEL.ROI_RNN_HEAD.RETURN_ATTENTION,
        )

        if self.training:
            out_dict['edge_logits'] = edge_logits
            out_dict['vertex_logits'] = vertex_logits

            if poly_class is not None:
                out_dict['poly_class'] = poly_class.type(torch.long)
        # TODO: 测试部分未完成
        # else:
        #     if fp_beam_size != 1 or lstm_beam_size != 1:
        #         # Automatically means that this is in test mode
        #         # because of the assertion in the beginning
        #
        #         # comparison metric is of shape [batch_size * fp_beam_size * lstm_beam_size]
        #         if 'tool' in self.mode:
        #             # Check for intersections and remove
        #             isect = utils.count_self_intersection(
        #                 out_dict['pred_polys'].cpu().numpy(),
        #                 self.encoder.feat_size
        #             )
        #             isect[isect != 0] -= float('inf')
        #             # 0 means no intersection, -inf for intersection
        #             isect = torch.from_numpy(isect).to(torch.float32).to(device)
        #             comparison_metric = comparison_metric + isect
        #             print (comparison_metric)
        #
        #         comparison_metric = comparison_metric.view(batch_size, fp_beam_size, lstm_beam_size)
        #         out_dict['pred_polys'] = out_dict['pred_polys'].view(batch_size, fp_beam_size, lstm_beam_size, -1)
        #
        #         # Max across beams
        #         comparison_metric, beam_idx = torch.max(comparison_metric, dim=-1)
        #
        #         # Max across first points
        #         comparison_metric, fp_beam_idx = torch.max(comparison_metric, dim=-1)
        #
        #         pred_polys = torch.zeros(batch_size, self.opts['max_poly_len'], device=device,
        #             dtype=out_dict['pred_polys'].dtype)
        #
        #         for b in torch.arange(batch_size, dtype=torch.int32):
        #             # Get best beam from all first points and all beams
        #             pred_polys[b, :] = out_dict['pred_polys'][b, fp_beam_idx[b], beam_idx[b, fp_beam_idx[b]], :]
        #
        #         out_dict['pred_polys'] = pred_polys

        out_dict['proposals'] = poly_class
        out_dict.pop('rnn_state')
        out_dict.pop('feats')

        if self.training:
            device = out_dict['poly_class'].device
            dt_targets = dt_targets_from_class(out_dict['poly_class'].cpu().numpy(),
                                               self.cfg.MODEL.ROI_RNN_HEAD.POOLER_RESOLUTION,
                                               self.cfg.MODEL.ROI_RNN_HEAD.DT_THRESOLD)
            fp_weight = self.cfg.MODEL.ROI_RNN_HEAD.FP_WEIGHT
            vertex_loss = losses.poly_vertex_loss_mle(torch.from_numpy(dt_targets).to(device),
                                               poly_masks, out_dict['logits'])
            fp_edge_loss =  fp_weight * losses.fp_edge_loss(edge_masks,
                                                                out_dict['edge_logits'])
            fp_vertex_loss = fp_weight * losses.fp_vertex_loss(ver_masks,
                                                                        out_dict['vertex_logits'])

        return out_dict, all_proposals, dict(vertex_loss = vertex_loss, loss_fp_edge = fp_edge_loss, loss_fp_vertex = fp_vertex_loss)


def build_roi_rnn_head(cfg,in_channels):
    return ROIRnnHead(cfg, in_channels)


