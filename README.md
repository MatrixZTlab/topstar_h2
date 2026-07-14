# topstar_h2

H2 humanoid robot — MuJoCo simulation runtime and RL policy runner.

This repo contains the H2 robot model (`h2_model/`) and the prebuilt binaries +
Python runner needed for the 3-terminal MuJoCo test. It extends
[topstar_rl_lab](https://github.com/MatrixZTlab/topstar_rl_lab), which provides
the `topstar_mujoco` simulator and policy training.

## Prerequisites

- Ubuntu 22.04, x86_64 (binaries in `build_sim/` are prebuilt for this platform)
- CycloneDDS runtime `libddsc.so.0` and its iceoryx dependencies — easiest via
  ROS 2 Humble (`source /opt/ros/humble/setup.bash` before running, so
  `LD_LIBRARY_PATH` resolves them)
- [topstar_mujoco](https://github.com/MatrixZTlab/topstar_mujoco) cloned and
  built in a **sibling directory** of this repo:

  ```bash
  git clone https://github.com/MatrixZTlab/topstar_mujoco ../topstar_mujoco
  cd ../topstar_mujoco/simulate && mkdir -p build && cd build && cmake .. && make
  ```
- Python: `pip install numpy pyyaml onnxruntime`

## 3-Terminal MuJoCo Test

```
Terminal 1: topstar_mujoco             ← MuJoCo physics + viewer
    │  DDS on loopback (rt/lowstate ↑, rt/lowcmd ↓)
    ▼
Terminal 2: mujoco_sim_bridge_v2       ← Shared memory, FSM, DDS services
    │  Shared memory (/ec_motor_shm)
    ▼
Terminal 3: h2_amp_rl_runner.py        ← ONNX policy inference
```

**Terminal 1** — MuJoCo viewer:

```bash
./run_mujoco_viewer.sh
```

**Terminal 2** — Simulation bridge (standalone mode: owns shared memory, runs
FSM, gait generator, and DDS service servers):

```bash
./run_mujoco_sim.sh
```

**Terminal 3** — Stand up, then run the AMP RL policy:

```bash
./run_topstar_client.sh --network_interface=lo --stand_up
# wait ~2 s for the stand-up transition to finish
python3 python/h2_amp_rl_runner.py
```

The runner loads `dist/amp_v2/policy.onnx` + `dist/amp_v2/deploy.yaml` by
default; override with `--policy-onnx` / `--deploy-yaml`. Use `--no-joystick
--auto-ai` to enter AI mode without a controller, or press **START** on a
PS2/PS4 controller. See `python3 python/h2_amp_rl_runner.py --help` for
velocity limits, gain scaling, and filtering options.

## Contents

| Path | Description |
|---|---|
| `h2_model/` | URDF/MJCF robot model, meshes, config |
| `build_sim/bin/mujoco_sim_bridge_v2` | Simulation bridge (prebuilt, x86_64) |
| `build_sim/bin/h2_loco_client_test` | DDS loco client for stand-up/FSM commands |
| `build_sim/lib/` | `libec_shared_mem_v2` shared-memory library |
| `python/h2_amp_rl_runner.py` | AMP ONNX policy runner (reads `/ec_motor_shm`) |
| `python/h2_rl_runner.py`, `python/h2_shm.py` | Runner support modules |
| `dist/amp_v2/` | Deployed AMP policy bundle (`policy.onnx`, `deploy.yaml`) |
