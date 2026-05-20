"""Distribution-aware attention-compression utilities and LM surgery primitives."""

from attention_compression.qk_surgery import (
    MultiHeadQKLowRankProjection,
    evaluate_loss,
    layer_is_complete,
    load_layer_qk_states,
    materialize_dense_linear_from_branch_states,
    patch_layer_qk_dense_v,
)

__all__ = [
    "counts",
    "windows",
    "MultiHeadQKLowRankProjection",
    "evaluate_loss",
    "layer_is_complete",
    "load_layer_qk_states",
    "materialize_dense_linear_from_branch_states",
    "patch_layer_qk_dense_v",
]
