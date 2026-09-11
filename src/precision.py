"""Opt-in mixed precision for MPS/CUDA training.

Ultralytics enables AMP on CUDA only: `check_amp` returns False for CPU/MPS, the
trainer hardcodes a CUDA GradScaler, and its `autocast()` helper defaults to
`device="cuda"`. This module activates the equivalent training loop on other
devices without editing the installed package (ultralytics 8.3.211, torch 2.14).

Design notes:
- `BaseTrainer._do_train` calls the module-global `autocast(self.amp)`; patching
  `ultralytics.engine.trainer.autocast` redirects the forward pass (and loss)
  into `torch.amp.autocast(device, dtype)`.
- `trainer.amp` is deliberately left False so training-time validation stays in
  fp32: on MPS, low-precision validation diverged from standalone evaluation
  (ultralytics#24399, never fixed upstream).
- fp16 needs an enabled `GradScaler(device)` against underflow; bf16 needs no
  scaling, so the trainer's default disabled scaler is exactly right.
"""

import torch

PRECISION_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_device(device):
    """Resolve a configured device, preferring MPS on Apple hosts when unspecified."""
    if device not in (None, ""):
        text = str(device)
        return torch.device(f"cuda:{text}" if text.isdigit() else text)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_mixed_precision(model, precision, device):
    """Enable autocast training on `model`. No-op for fp32."""
    if precision == "fp32":
        return
    if precision not in PRECISION_DTYPES:
        raise ValueError(f"Unknown training precision: {precision!r}.")
    dtype = PRECISION_DTYPES[precision]
    device = resolve_device(device)
    if device.type == "cpu":
        raise ValueError("Mixed precision training requires an MPS or CUDA device.")
    device_type = device.type

    import ultralytics.engine.trainer as trainer_module

    def autocast(_enabled):
        # The trainer passes its (False) amp flag; mixed precision is on whenever
        # this helper is installed, so the flag is intentionally ignored.
        return torch.amp.autocast(device_type, dtype=dtype, enabled=True)

    trainer_module.autocast = autocast

    def configure(trainer):
        if dtype == torch.float16:
            trainer.scaler = torch.amp.GradScaler(device_type, enabled=True)

    model.add_callback("on_train_start", configure)
