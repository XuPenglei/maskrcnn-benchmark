import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

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