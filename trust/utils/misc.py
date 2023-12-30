import torch
import sys
import os
from typing import Optional


from torch_scatter import scatter_sum, scatter_max

from torch_scatter.utils import broadcast

# Adapted from https://github.com/rusty1s/pytorch_scatter
# def scatter_logsumexp(dim, index, src):
#     output_shape = src.shape[:dim] + (torch.max(index), ) + src.shape[dim + 1:]

#     max_per_group = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
#     max_per_group = max_per_group.scatter_reduce_(dim, index, src, reduce="amax", include_self=False)
#     max_per_group_expanded = max_per_group.gather(dim, index)

#     src_norm = src - max_per_group_expanded
#     src_norm_exp = src_norm.exp()
#     src_norm_sum_exp = max_per_group.scatter_reduce_(dim, index, src_norm_exp, reduce="sum", include_self=False)
#     src_norm_log_sum_exp = src_norm_sum_exp.log()
#     src_log_sum_exp = src_norm_log_sum_exp + max_per_group

#     return src_log_sum_exp

# Corrected version which handles -inf correctly.
def scatter_logsumexp(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                      out: Optional[torch.Tensor] = None,
                      dim_size: Optional[int] = None,
                      eps: float = 1e-12) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_logsumexp` can only be computed over '
                         'tensors with floating point data types.')

    index = broadcast(index, src, dim)

    if out is not None:
        dim_size = out.size(dim)
    else:
        if dim_size is None:
            dim_size = int(index.max()) + 1

    size = list(src.size())
    size[dim] = dim_size
    max_value_per_index = torch.full(size, float('-inf'), dtype=src.dtype,
                                     device=src.device)
    scatter_max(src, index, dim, max_value_per_index, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_score = src - max_per_src_element
    recentered_score.masked_fill_(torch.isnan(recentered_score), float('-inf'))

    if out is not None:
        out = out.sub_(max_value_per_index).exp_()

    sum_per_index = scatter_sum(recentered_score.exp_(), index, dim, out,
                                dim_size)

    out = sum_per_index.add_(eps).log_().add_(max_value_per_index)
    return out.nan_to_num_()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

