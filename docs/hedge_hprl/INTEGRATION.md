# HPRL Clean Mainline Integration

The HPRL runtime boundary is intentionally narrow:

1. historical market tensors enter `VectorizedHedgeEnv` on the configured CPU or CUDA device;
2. a learner produces continuous independent LONG/SHORT exposures on the same training device;
3. `HedgeActionProjector` applies hard per-leg, gross, net and step-change limits;
4. replay stores the executed/projected action and can reside on the training device or CPU;
5. `CleanMainlineSignalAdapter` converts projected targets to the canonical `SignalSnapshot`;
6. existing Clean Mainline planning, risk and execution code remains authoritative.

The device authority is `HPRLTrainingConfig.device`. The default `auto` setting resolves to CUDA
when available, otherwise CPU. Explicit `cuda` or `cuda:N` fails closed if that CUDA runtime/device
is unavailable; it does not silently fall back to CPU. `replay_device="same"` is the preferred GPU
throughput profile, while `replay_device="cpu"` is available to trade throughput for lower VRAM
usage.

Performance runs normally leave `runtime_checks=false`, because repeated tensor finite/range checks
can synchronize CUDA. Standalone components remain strict by default when constructed directly, and
the test suite exercises those validation paths. `metrics_interval` similarly controls how often
training metrics are materialized as host scalars.

The bridge does not import or call the existing FreqAI RL implementation. No HPRL module contains
Binance credentials, CCXT order methods or direct exchange write APIs.
