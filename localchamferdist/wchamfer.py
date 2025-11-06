import warnings
from collections import namedtuple
from typing import Optional, Union

import torch

# from localchamferdist import _C
from torch.autograd import Function
from localchamferdist.chamfer import knn_points


class WeightedChamferDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        weights_source: Optional[torch.Tensor] = None,
        bidirectional: Optional[bool] = False,
        reverse: Optional[bool] = False,
        batch_reduction: Optional[str] = "mean",
        point_reduction: Optional[str] = "sum",
    ):
        
        if not isinstance(source_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(pts))
            )
        if not isinstance(target_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(pts))
            )
        if source_cloud.device != target_cloud.device:
            raise ValueError(
                "Source and target clouds must be on the same device. "
                f"Got {source_cloud.device} and {target_cloud.device}."
            )

        if point_reduction != "sum" and point_reduction != "mean" and point_reduction != None:
            raise ValueError('Point reduction must either be "sum" or "mean" or None.')
        if batch_reduction != "sum" and batch_reduction != "mean" and batch_reduction != None:
            raise ValueError('Batch reduction must either be "sum" or "mean" or None.')
        
        batchsize_source, lengths_source, dim_source = source_cloud.shape
        batchsize_target, lengths_target, dim_target = target_cloud.shape

        lengths_source = (
            torch.ones(batchsize_source, dtype=torch.long, device=source_cloud.device)
            * lengths_source
        )
        lengths_target = (
            torch.ones(batchsize_target, dtype=torch.long, device=target_cloud.device)
            * lengths_target
        )
        
        source_nn = knn_points(
            source_cloud,
            target_cloud,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1
        )

        target_nn = None
        if reverse or bidirectional:
            target_nn = knn_points(
                target_cloud,
                source_cloud,
                lengths1=lengths_target,
                lengths2=lengths_source,
                K=1
            )

        chamfer_forward = source_nn.dists[..., 0]
        chamfer_backward = None

        # Apply weights per-vertex
        # Note that only the source have weights in this implementation
        if weights_source is not None:
            chamfer_forward = chamfer_forward * weights_source

        if reverse or bidirectional:
            chamfer_backward = target_nn.dists[..., 0]
            if weights_source is not None:
                chamfer_backward = chamfer_backward * weights_source

        # Reduce
        if reverse or bidirectional:
            # Backward Chamfer distance (batchsize_source, lengths_source)
            chamfer_backward = target_nn.dists[..., 0]

        if point_reduction == "sum":
            chamfer_forward = chamfer_forward.sum(1)  # (batchsize_source,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.sum(1)  # (batchsize_target,)
        elif point_reduction == "mean":
            chamfer_forward = chamfer_forward.mean(1)  # (batchsize_source,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.mean(1)  # (batchsize_target,)

        if batch_reduction == "sum":
            chamfer_forward = chamfer_forward.sum()  # (1,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.sum()  # (1,)
        elif batch_reduction == "mean":
            chamfer_forward = chamfer_forward.mean()  # (1,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.mean()  # (1,)

        if bidirectional:
            return chamfer_forward + chamfer_backward
        if reverse:
            return chamfer_backward

        return chamfer_forward
