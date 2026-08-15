"""Device, precision, and accelerator policy for the isolated HPRL subsystem."""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass

from .errors import HPRLConfigError, HPRLDependencyError


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only exercised without optional dependency
        raise HPRLDependencyError(
            "HPRL requires the project Torch dependency. Install the Clean Mainline RL "
            "dependencies into the project-local .venv."
        ) from exc
    return torch


def _normalized_device_request(requested: str | object) -> str:
    value = str(requested).strip().lower()
    if value == "gpu":
        return "cuda"
    if value in {"auto", "cpu", "cuda"}:
        return value
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return value
    raise HPRLConfigError("device must be one of auto/cpu/cuda/gpu/cuda:<index>")


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    requested: str
    resolved: str
    cuda_available: bool
    device_name: str
    cuda_device_count: int = 0
    cuda_index: int | None = None
    total_memory_bytes: int | None = None
    bf16_supported: bool = False
    tf32_supported: bool = False


def resolve_device(requested: str | object = "auto") -> DeviceInfo:
    torch = require_torch()
    normalized = _normalized_device_request(requested)
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0

    if normalized == "auto":
        normalized = "cuda:0" if cuda_available else "cpu"
    elif normalized in {"cuda", "gpu"}:
        normalized = "cuda:0"

    if normalized == "cpu":
        return DeviceInfo(
            requested=str(requested),
            resolved="cpu",
            cuda_available=cuda_available,
            device_name="CPU",
            cuda_device_count=cuda_count,
        )

    if not cuda_available:
        raise HPRLDependencyError(
            f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled project Torch build and a compatible NVIDIA driver, or use "
            "device='cpu'/'auto'."
        )
    index = int(normalized.split(":", 1)[1])
    if index < 0 or index >= cuda_count:
        raise HPRLConfigError(
            f"CUDA device index {index} is outside the available range 0..{cuda_count - 1}"
        )
    props = torch.cuda.get_device_properties(index)
    # Prefer device properties for non-current GPUs so merely inspecting cuda:N does not change
    # global CUDA state. Current-device helpers remain useful when capability fields are absent.
    major = int(getattr(props, "major", 0))
    if index == torch.cuda.current_device():
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        tf32_fn = getattr(torch.cuda, "is_tf32_supported", None)
        tf32_supported = bool(tf32_fn()) if callable(tf32_fn) else major >= 8
    else:
        bf16_supported = major >= 8
        tf32_supported = major >= 8
    return DeviceInfo(
        requested=str(requested),
        resolved=f"cuda:{index}",
        cuda_available=True,
        device_name=str(props.name),
        cuda_device_count=cuda_count,
        cuda_index=index,
        total_memory_bytes=int(props.total_memory),
        bf16_supported=bf16_supported,
        tf32_supported=tf32_supported,
    )


def torch_device(requested: str | object = "auto"):
    return require_torch().device(resolve_device(requested).resolved)


def configured_torch_device(configured: str | object, requested: str | object):
    configured_info = resolve_device(configured)
    requested_info = resolve_device(requested)
    if configured_info.resolved != requested_info.resolved:
        raise HPRLConfigError(
            "explicit runtime device does not match the HPRL training configuration: "
            f"configured={configured_info.resolved}, requested={requested_info.resolved}"
        )
    return require_torch().device(requested_info.resolved)


def cpu_torch_device(requested: str | object = "cpu"):
    """Compatibility alias retained for early HPRL callers.

    The name is historical.  New code should use :func:`torch_device`; this alias now resolves the
    full cpu/cuda/auto policy so older HPRL imports do not force a CPU fallback.
    """
    return torch_device(requested)


def configure_acceleration(
    device: str | object,
    *,
    deterministic: bool = False,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = False,
    cuda_memory_fraction: float | None = None,
) -> None:
    torch = require_torch()
    resolved = torch_device(device)
    if matmul_precision not in {"highest", "high", "medium"}:
        raise HPRLConfigError("matmul_precision must be highest/high/medium")
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    torch.set_float32_matmul_precision(matmul_precision)
    if resolved.type != "cuda":
        return
    torch.cuda.set_device(resolved)
    # These flags remain supported by the project's pinned Torch and map to TF32-backed FP32
    # matmul/convolution acceleration on compatible NVIDIA GPUs.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not deterministic)
    if cuda_memory_fraction is not None:
        fraction = float(cuda_memory_fraction)
        if not 0.0 < fraction <= 1.0:
            raise HPRLConfigError("cuda_memory_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device=resolved)


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
    device: str | object | None = None,
) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    torch = require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if device is not None and torch_device(device).type == "cuda":
        torch.cuda.set_device(torch_device(device))


class PrecisionManager:
    """CUDA automatic-mixed-precision helper with an FP32-stable CPU path."""

    def __init__(
        self,
        device: str | object,
        *,
        enabled: bool = False,
        dtype: str = "auto",
        grad_clip_foreach: bool = True,
    ) -> None:
        torch = require_torch()
        self.torch = torch
        self.device = torch_device(device)
        self.enabled = bool(enabled)
        self.grad_clip_foreach = bool(grad_clip_foreach)
        normalized = str(dtype).strip().lower()
        if normalized not in {"auto", "float16", "bfloat16"}:
            raise HPRLConfigError("amp_dtype must be auto/float16/bfloat16")
        if self.enabled and self.device.type != "cuda":
            raise HPRLConfigError("mixed_precision currently requires a CUDA training device")

        self.dtype_name = "float32"
        self.autocast_dtype = torch.float32
        self.scaler = None
        if self.enabled:
            with torch.cuda.device(self.device):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
            if normalized == "auto":
                normalized = "bfloat16" if bf16_supported else "float16"
            if normalized == "bfloat16" and not bf16_supported:
                raise HPRLDependencyError(
                    "bfloat16 AMP was requested but the CUDA device lacks BF16 support"
                )
            self.dtype_name = normalized
            self.autocast_dtype = torch.bfloat16 if normalized == "bfloat16" else torch.float16
            # BF16 has FP32-like exponent range and normally does not need gradient scaling.
            if normalized == "float16":
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return self.torch.amp.autocast(
            "cuda",
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def backward_and_clip(self, loss, optimizer, parameters, max_norm: float):
        """Run backward + unscale/foreach clipping while leaving optimizer.step separate.

        The split is used by the XQC decomposition profiler and pre-bound optimizer plans.
        It preserves the original scaler ordering exactly.
        """
        params = parameters if isinstance(parameters, tuple) else tuple(parameters)
        optimizer.zero_grad(set_to_none=True)
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
        norm = self.torch.nn.utils.clip_grad_norm_(
            params, max_norm, foreach=self.grad_clip_foreach
        )
        return norm.detach()

    def optimizer_step(self, optimizer) -> None:
        if self.scaler is None:
            optimizer.step()
            return
        self.scaler.step(optimizer)
        self.scaler.update()

    def backward_step(self, loss, optimizer, parameters, max_norm: float) -> float:
        norm = self.backward_and_clip(loss, optimizer, parameters, max_norm)
        self.optimizer_step(optimizer)
        return norm


def synchronize_device(device: str | object) -> None:
    torch = require_torch()
    resolved = torch_device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def cuda_memory_stats(device: str | object) -> dict[str, int]:
    torch = require_torch()
    resolved = torch_device(device)
    if resolved.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
        "reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
    }
