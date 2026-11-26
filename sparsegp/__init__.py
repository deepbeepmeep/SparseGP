"""SparseGP forward-only Sol³ and Sparge attention kernels."""

from .core import backend_for, dense_attention, sol3_attention, sparge_attention, block_sparse_sage2_attn_cuda

__version__ = "1.2.0"
__all__ = ["backend_for", "dense_attention", "sol3_attention", "sparge_attention", "block_sparse_sage2_attn_cuda"]
