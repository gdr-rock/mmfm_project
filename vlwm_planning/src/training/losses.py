from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # allow non-training environments to import module
    torch = None
    F = None


IGNORE_INDEX = -100


def autoregressive_ce_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    ignore_index: int = IGNORE_INDEX,
) -> "torch.Tensor":
    """
    System-1 Eq. (1): token-wise autoregressive CE
    L = -sum_t log p(x_t | x_<t, context)

    Expects:
    - logits: [B, T, V]
    - labels: [B, T] where non-target/context positions are ignore_index

    This function performs the standard causal LM shift internally.
    """
    if torch is None or F is None:
        raise ModuleNotFoundError("torch is required for loss computation")

    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, V], got shape {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels [B, T], got shape {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Batch/sequence mismatch: logits {tuple(logits.shape[:2])} vs labels {tuple(labels.shape)}"
        )

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
