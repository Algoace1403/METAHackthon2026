"""Tiny shared GPU-capability helpers.

Single source of truth for "should we use bf16 or fp16 on this device?"
Referenced by both ``train_sft`` and ``evaluate_sft`` so the rule can be
updated in one place.

Why it matters: Turing GPUs (T4, sm_7.5) do not have native bf16 Tensor
Cores. Setting ``bf16=True`` on a T4 either falls back to FP32 math (slow)
or is rejected outright by flash-attention kernels. Ampere (A100, sm_8.0)
and later have native bf16.
"""

from __future__ import annotations


def preferred_dtype_flags() -> tuple[bool, bool]:
    """Return ``(use_bf16, use_fp16)`` flags suitable for TRL's SFTConfig.

    Falls back to FP32 (both False) if CUDA is unavailable — the caller is
    expected to skip training in that case anyway.
    """
    try:
        import torch
    except ImportError:
        return (False, False)
    if not torch.cuda.is_available():
        return (False, False)
    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:  # Ampere, Ada, Hopper, Blackwell
        return (True, False)
    return (False, True)  # Turing, Volta, Pascal — FP16 path


def preferred_torch_dtype():
    """Return the matching ``torch.dtype`` for model load-time precision.

    Ampere+ → ``torch.bfloat16``; older Turing etc → ``torch.float16``.
    Raises ``ImportError`` only if the caller imports torch later; here we
    return ``None`` if CUDA is absent so eval harnesses can fall through.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16
