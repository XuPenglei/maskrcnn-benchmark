import torch
import torch.nn.functional as F


def fp_edge_loss(gt_edges, edge_logits):
    """
    Edge loss in the first point network

    gt_edges: [batch_size, grid_size, grid_size] of 0/1
    edge_logits: [batch_size, grid_size*grid_size]
    """
    edges_shape = gt_edges.size()
    gt_edges = gt_edges.view(edges_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges)

    return torch.mean(loss)


def fp_vertex_loss(gt_verts, vertex_logits):
    """
    Vertex loss in the first point network

    gt_verts: [batch_size, grid_size, grid_size] of 0/1
    vertex_logits: [batch_size, grid_size**2]
    """
    verts_shape = gt_verts.size()
    gt_verts = gt_verts.view(verts_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(vertex_logits, gt_verts)

    return torch.mean(loss)


def poly_vertex_loss_mle(targets, mask, logits):
    """
    Classification loss for polygon vertex prediction

    targets: [batch_size, time_steps, grid_size**2+1]
    Each element is y*grid_size + x, or grid_size**2 for EOS
    mask: [batch_size, time_steps]
    Mask stipulates whether this time step is used for training
    logits: [batch_size, time_steps, grid_size**2 + 1]
    """
    batch_size = mask.size(0)

    # Remove the zeroth time step
    logits = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))  # (batch*(time_steps-1), grid_size**2 + 1)
    targets = targets[:, 1:, :].contiguous().view(-1, logits.size(-1))  # (batch*(time_steps-1))
    mask = mask[:, 1:].contiguous().view(-1)  # (batch*(time_steps-1))

    # Cross entropy between targets and softmax outputs
    loss = torch.sum(-targets * F.log_softmax(logits, dim=1), dim=1)
    loss = loss * mask.type_as(loss)
    loss = loss.view(batch_size, -1)
    # Sum across time
    loss = torch.sum(loss, dim=1)
    # Mean across batches
    return torch.mean(loss)
