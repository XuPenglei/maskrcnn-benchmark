import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstVertex(nn.Module):
    def __init__(self, cfg):
        """ max_len is the max length of vertexes list allowed """
        super(FirstVertex, self).__init__()
        self.grid_size = cfg.MODEL.ROI_RNN_HEAD.POOLER_RESOLUTION
        self.max_len = cfg.MODEL.ROI_RNN_HEAD.MAX_LEN
        feats_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.edge_conv = nn.Conv2d(
            in_channels=feats_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.edge_fc = nn.Linear(
            in_features=grid_size ** 2 * 16,
            out_features=grid_size ** 2
        )

        self.vertex_conv = nn.Conv2d(
            in_channels=feats_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.vertex_fc = nn.Linear(
            in_features=grid_size ** 2 * 16,
            out_features=grid_size ** 2
        )

    def forward(self, feats, temperature=0.0, beam_size=1):
        """
        if temperature < 0.01, use greedy
        else, use temperature
        """
        batch_size = feats.size(0)
        conv_edge = self.edge_conv(feats)
        conv_edge = F.relu(conv_edge, inplace=True)
        edge_logits = self.edge_fc(conv_edge.view(batch_size, -1))

        # Different from before, this used to take conv_edge as input before
        conv_vertex = self.vertex_conv(feats)
        conv_vertex = F.relu(conv_vertex)
        vertex_logits = self.vertex_fc(conv_vertex.view(batch_size, -1))
        logprobs = F.log_softmax(vertex_logits, -1)

        # Sample a first vertex
        if temperature < 0.01:
            logprob, pred_first = torch.topk(logprobs, beam_size, dim=-1)

        else:
            probs = torch.exp(logprobs / temperature)
            pred_first = torch.multinomial(probs, beam_size)

            # Get logprob of the sampled vertex
            logprob = logprobs.gather(1, pred_first)

        # Remove the last dimension if it is 1
        pred_first = torch.squeeze(pred_first, dim=-1)
        logprob = torch.squeeze(logprob, dim=-1)

        return edge_logits, vertex_logits, logprob, pred_first

