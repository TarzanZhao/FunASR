"""Measurement hooks for the torch-performance-agent campaign. Every entry point is a
no-op unless its env var is set, so the timed run pays nothing.

  PROBE=1 PROBE_OUT=<file>.json   correctness recorder (package `probe`): loss, acc, shapes,
                                  logits stats, grad/param norm, val loss, paired by tag.
  TPA_PROFILE=1                   torch.profiler over a window of steps:
    TPA_PROFILE_WAIT=<n> TPA_PROFILE_ACTIVE=<n> TPA_PROFILE_OUT=<dir>  -> <dir>/trace.rank<r>.json
  TPA_STEPTIMES_OUT=<file>.json   per-step wall (perf_counter, from the trainer's own timers),
                                  max_memory_allocated/reserved, and host RSS, written per step.
"""
from __future__ import annotations

import json
import os
import resource
import time

import torch

try:
    import probe
except ImportError:  # the recorder is optional; timed runs never need it
    probe = None

STEP = 0  # 1-based, set by on_step_begin; model.py reads it for its tags
_NORM_STEPS = {1, 2, 10, 50, 100, 200, 374}


def enabled() -> bool:
    return probe is not None and probe.enabled()


def on_step_begin(step: int) -> None:
    global STEP
    STEP = step


def probe_batch(batch: dict) -> None:
    if not enabled():
        return
    for k in ("speech", "speech_lengths", "input_ids", "attention_mask"):
        v = batch.get(k)
        if torch.is_tensor(v):
            for i, d in enumerate(v.shape):
                probe.record(f"shape_{k}{i}_step{STEP}", int(d))
    if torch.is_tensor(batch.get("speech_lengths")):
        probe.record(f"speech_frames_step{STEP}", int(batch["speech_lengths"].sum().item()))


def probe_forward(loss_dict: dict) -> None:
    if not enabled():
        return
    probe.record(f"loss_step{STEP}", float(loss_dict["loss"].detach().float().item()), rtol=0.0, atol=1e-2)
    acc = loss_dict["stats"].get("acc")
    if acc is not None:
        probe.record(f"acc_step{STEP}", float(torch.as_tensor(acc).float().item()), rtol=0.0, atol=1e-2)


def probe_logits(logits: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor) -> None:
    """Called from the model forward. rtol 1e-2 is the bf16 noise floor (probe README)."""
    if not enabled() or STEP == 0:
        return
    with torch.no_grad():
        lf = logits.detach().float()
        probe.record(f"logits_absmean_step{STEP}", float(lf.abs().mean().item()), rtol=1e-2)
        probe.record(f"logits_std_step{STEP}", float(lf.std().item()), rtol=1e-2)
        mask = labels[:, 1:] != -100
        n_correct = int(((preds[:, :-1] == labels[:, 1:]) & mask).sum().item())
        probe.record(f"n_correct_step{STEP}", n_correct, rtol=1e-2)
        probe.record(f"n_labels_step{STEP}", int(mask.sum().item()))


def probe_update(model, grad_norm) -> None:
    if not enabled() or STEP not in _NORM_STEPS:
        return
    if grad_norm is not None:
        probe.record(f"grad_norm_step{STEP}", float(torch.as_tensor(grad_norm).float().item()), rtol=1e-2)
    with torch.no_grad():
        sq = torch.zeros((), dtype=torch.float64, device="cpu")
        for p in model.parameters():
            if p.requires_grad:
                sq += p.detach().float().pow(2).sum().double().cpu()
        probe.record(f"param_norm_step{STEP}", float(sq.sqrt().item()), rtol=1e-4)


def probe_val(val_loss: float, val_acc: float) -> None:
    if not enabled():
        return
    probe.record("val_loss", float(val_loss), rtol=0.0, atol=1e-3)
    probe.record("val_acc", float(val_acc), rtol=0.0, atol=1e-2)
    probe.flush()


# ---------------------------------------------------------------- profiler

class Profiler:
    """torch.profiler over TPA_PROFILE_WAIT + TPA_PROFILE_ACTIVE steps; exports one chrome trace."""

    def __init__(self, rank: int):
        self.on = os.environ.get("TPA_PROFILE") == "1"
        self.prof = None
        if not self.on:
            return
        wait = int(os.environ.get("TPA_PROFILE_WAIT", "3"))
        active = int(os.environ.get("TPA_PROFILE_ACTIVE", "3"))
        self.out = os.environ.get("TPA_PROFILE_OUT", ".")
        os.makedirs(self.out, exist_ok=True)
        self.path = os.path.join(self.out, f"trace.rank{rank}.json")
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=1, active=active, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(self.path),
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        self.prof.start()
        self.window = (wait + 2, wait + 1 + active)  # 1-based steps captured

    def step(self) -> None:
        if self.prof is not None:
            self.prof.step()

    def stop(self) -> None:
        if self.prof is not None:
            self.prof.stop()
            self.prof = None


class StepTimes:
    """Per-step wall from the trainer's own timers plus memory, to TPA_STEPTIMES_OUT."""

    def __init__(self):
        self.path = os.environ.get("TPA_STEPTIMES_OUT")
        self.rows = []
        self.t0 = time.time()

    def record(self, step: int, speed_stats: dict) -> None:
        if not self.path:
            return
        self.rows.append({
            "step": step,
            **{k: float(v) for k, v in speed_stats.items()},
            "t_since_epoch_start_s": time.time() - self.t0,
            "gpu_alloc_gb": torch.cuda.memory_allocated() / 2**30,
            "gpu_max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "host_maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        })
        if step % 20 == 0 or step < 5:
            self.flush()

    def flush(self) -> None:
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.rows, f)
