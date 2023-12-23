import torch
import sys
import os

# Adapted from https://github.com/rusty1s/pytorch_scatter
def scatter_logsumexp(dim, index, src):
    output_shape = src.shape[:dim] + (torch.max(index), ) + src.shape[dim + 1:]

    max_per_group = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
    max_per_group = max_per_group.scatter_reduce_(dim, index, src, reduce="amax", include_self=False)
    max_per_group_expanded = max_per_group.gather(dim, index)

    src_norm = src - max_per_group_expanded
    src_norm_exp = src_norm.exp()
    src_norm_sum_exp = max_per_group.scatter_reduce_(dim, index, src_norm_exp, reduce="sum", include_self=False)
    src_norm_log_sum_exp = src_norm_sum_exp.log()
    src_log_sum_exp = src_norm_log_sum_exp + max_per_group

    return src_log_sum_exp

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

