"""Low-overhead stage attribution helpers for diagnostic-only HPRL profiling."""

from __future__ import annotations

from contextlib import contextmanager
import time

from .device import require_torch, torch_device


torch = require_torch()


class StageRecorder:
    """Record named diagnostic stages with CUDA Events/NVTX or CPU wall time.

    CUDA event synchronization is intentional here: this recorder is never used by the production
    throughput path.  Keeping attribution in a separate diagnostic pass prevents instrumentation
    from destroying replay/H2D/update overlap in the real benchmark.
    """

    def __init__(self, device) -> None:
        self.device = torch_device(device)
        self._seconds: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def record(self, name: str, *, domain: str = "auto"):
        label = str(name)
        normalized = str(domain).strip().lower()
        if normalized not in {"auto", "host", "cuda"}:
            raise ValueError("stage timing domain must be auto/host/cuda")
        use_cuda = self.device.type == "cuda" and normalized != "host"
        if not use_cuda:
            started = time.perf_counter()
            try:
                yield
            finally:
                self._add(label, time.perf_counter() - started)
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx = getattr(torch.cuda, "nvtx", None)
        pushed = False
        if nvtx is not None:
            try:
                nvtx.range_push(f"HPRL::{label}")
                pushed = True
            except Exception:
                pushed = False
        start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream(self.device))
            end.synchronize()
            if pushed:
                try:
                    nvtx.range_pop()
                except Exception:
                    pass
            self._add(label, max(0.0, float(start.elapsed_time(end))) / 1000.0)

    def _add(self, name: str, seconds: float) -> None:
        self._seconds[name] = self._seconds.get(name, 0.0) + float(seconds)
        self._counts[name] = self._counts.get(name, 0) + 1

    def summary(self) -> dict[str, object]:
        stages = {
            name: {
                "seconds": seconds,
                "milliseconds": seconds * 1000.0,
                "count": self._counts.get(name, 0),
                "mean_milliseconds": (
                    seconds * 1000.0 / max(1, self._counts.get(name, 0))
                ),
            }
            for name, seconds in sorted(self._seconds.items())
        }
        total = sum(self._seconds.values())
        return {"total_seconds": total, "stages": stages}
