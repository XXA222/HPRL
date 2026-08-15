# HPRL Clean Mainline Architecture

HPRL is an independent high-performance reinforcement-learning subsystem under
`freqtrade/hedge/hprl/`. It is parallel to both `freqtrade/freqai/RL/` and
`freqtrade/freqai/hedge_rl/`; neither existing RL implementation is modified or imported.

HPRL supports `auto`, `cpu`, `cuda` and `cuda:N` execution. `auto` prefers CUDA when the project
Torch runtime reports a usable CUDA device and otherwise uses CPU. CUDA mode can keep the tensor
market environment, learner networks, reward state and Replay Buffer resident on the selected GPU.
Replay may instead be placed on CPU when VRAM is constrained. Optional CUDA acceleration includes
AMP (`float16` or `bfloat16`), TF32, configurable matrix-multiplication precision and periodic
metrics materialization to avoid unnecessary device synchronization.

HPRL emits projected LONG/SHORT target exposures and can bridge those targets to the canonical
`SignalSnapshot` contract. From that boundary, the existing Clean Mainline integration, planning,
risk and execution authorities remain in control. HPRL owns no exchange client and no live-order
write capability.

Algorithms currently implemented as project-native research adaptations are XQC, SimbaV2-SAC,
FastDSAC, FastTD3 and ReBRAC-v2. ReCAP-inspired regime composition and FineFT-inspired OOD routing
remain external policy layers rather than being fused into one learner.

`config_examples/hprl.example.json` demonstrates portable `auto` selection.
`config_examples/hprl.gpu.example.json` demonstrates an explicit GPU-oriented profile. A real
accelerator training-path check is available with:

```text
python -m freqtrade.hedge.hprl train-smoke --device cuda --algorithm fast_td3
```

Validation is performed by the normal Clean Mainline validators plus
`tools/validate_hprl_clean_mainline_200.py`. The root `CLEAN-MAINLINE-MANIFEST.json` remains
the only source hash authority.
