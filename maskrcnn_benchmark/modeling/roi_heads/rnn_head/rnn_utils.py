import torch
from torch.nn import functional as F

import numpy as np

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

from scipy.ndimage.morphology import distance_transform_cdt

def dt_targets_from_class(poly, grid_size, dt_threshold):
    """
    NOTE: numpy function!
    poly: [bs, time_steps], each value in [0, grid*size**2+1)
    grid_size: size of the grid the polygon is in
    dt_threshold: threshold for smoothing in dt targets

    returns:
    full_targets: [bs, time_steps, grid_size**2+1] array containing
    dt smoothed targets to be used for the polygon loss function
    """
    full_targets = []
    for b in range(poly.shape[0]):
        targets = []
        for p in poly[b]:
            t = np.zeros(grid_size**2+1, dtype=np.int32)
            t[p] += 1

            if p != grid_size**2:#EOS
                spatial_part = t[:-1]
                spatial_part = np.reshape(spatial_part, [grid_size, grid_size, 1])

                # Invert image
                spatial_part = -1 * (spatial_part - 1)
                # Compute distance transform
                spatial_part = distance_transform_cdt(spatial_part, metric='taxicab').astype(np.float32)
                # Threshold
                spatial_part = np.clip(spatial_part, 0, dt_threshold)
                # Normalize
                spatial_part /= dt_threshold
                # Invert back
                spatial_part = -1. * (spatial_part - 1.)

                spatial_part /= np.sum(spatial_part)
                spatial_part = spatial_part.flatten()

                t = np.concatenate([spatial_part, [0.]], axis=-1)

            targets.append(t.astype(np.float32))
        full_targets.append(targets)

    return np.array(full_targets, dtype=np.float32)

def class_to_grid(poly, out_tensor, grid_size):
    """
    NOTE: Torch function
    accepts out_tensor to do it inplace

    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    # Remove old state of out_tensor

    b = 0
    for i in poly:
        if i < grid_size * grid_size:
            x = (i % grid_size).long()
            y = (i / grid_size).long()
            out_tensor[b, 0, y, x] = 1
        b += 1

    return out_tensor


def xy_to_class(poly, grid_size):
    """
    NOTE: Torch function
    poly: [bs, time_steps, 2]

    Returns: [bs, time_steps] with class label
    for x,y location or EOS token
    """
    batch_size = poly.size(0)
    time_steps = poly.size(1)

    poly[:, :, 1] *= grid_size
    poly = torch.sum(poly, dim=-1)

    poly[poly < 0] = grid_size ** 2
    # EOS token

    return poly

def project_polys_on_boxes(segmentation_masks, proposals, discretization_size, max_len):
    """
    Given polygon like segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the polygons for them to be fed to rnn
    and loss evaluator.

    Arguments:
        segmentation_masks: an instance of SegmentationMask, must be mode "poly"
        proposals: an instance of BoxList
        max_len: max length of vertex list
    """
    ver_masks = []
    edge_masks = []
    arr_polys = []
    # poly_masks are not binary mask, it's for record valid vertexes
    poly_masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        rnn_tensors = scaled_mask.get_rnn_tensor(max_len)
        ver_masks.append(rnn_tensors[0])
        edge_masks.append(rnn_tensors[1])
        arr_polys.append(rnn_tensors[2])
        poly_masks.append(rnn_tensors[3])
    if len(ver_masks) == 0:
        ver_masks = torch.empty([0, M, M], dtype=torch.uint8)
        edge_masks = torch.empty([0, M, M], dtype=torch.uint8)
        arr_polys = torch.empty([0, max_len, 2], dtype=torch.int32)
        poly_masks = torch.empty([0, max_len], dtype=torch.int32)
        return ver_masks, edge_masks, arr_polys, poly_masks
    return torch.stack(ver_masks, dim=0).to(device, dtype=torch.float32), \
            torch.stack(edge_masks,dim=0).to(device, dtype= torch.float32), \
            torch.stack(arr_polys,dim=0).to(device, dtype= torch.float32), \
           torch.stack(poly_masks, dim=0).to(device, dtype=torch.float32)

class Target_Preprocessor(object):
    def __init__(self,cfg):
        self.cfg = cfg
        self.matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self,proposals,targets,enlarge_scale):
        labels = []
        ver_masks = []
        edge_masks = []
        arr_polys = []
        poly_masks = []
        out_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]
            if enlarge_scale > 0:
                positive_proposals = positive_proposals.enlarge(enlarge_scale)
            out_proposals.append(positive_proposals)

            rnn_input_per_img = project_polys_on_boxes(
                segmentation_masks, positive_proposals,
                self.cfg.MODEL.ROI_RNN_HEAD.POOLER_RESOLUTION,
                self.cfg.MODEL.ROI_RNN_HEAD.MAX_LEN
            )
            ver_masks.append(rnn_input_per_img[0])
            edge_masks.append(rnn_input_per_img[1])
            arr_polys.append(rnn_input_per_img[2])
            poly_masks.append(rnn_input_per_img[3])
        return cat(ver_masks,0),cat(edge_masks,0),\
               cat(arr_polys,0),cat(poly_masks,0),out_proposals




