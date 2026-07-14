"""H2 AMP RL Inference Runner — for gait-clock policies (v14_lsgan and later).

Derived from h2_rl_runner.py; reuses its DDS/joystick/safety machinery via
import. The one functional difference is the `gait_phase` observation term:
AMP policies trained with gait-clock rewards (v13+) condition on a periodic
phase signal

    [sin(2π·φ_L), cos(2π·φ_L), sin(2π·φ_R), cos(2π·φ_R), r_L, r_R]

where φ_{L,R} = (t/gait_cycle + offset_{L,R}) mod 1 advances with wall-clock
policy steps and r is the constant air ratio. h2_rl_runner.py returns zeros
for unknown obs terms, which is permanently out-of-distribution for these
policies (even sin=cos=0 is off the unit circle) — hence this runner, which
generates the clock and refuses to run if the deploy.yaml contains any obs
term it cannot compute.

Clock semantics match training (topstar_rl_lab mdp.gait_phase_obs):
  - t counts policy steps since AI-mode entry, held at 0 through warmup so
    the observation history at the first inference matches Isaac Lab's
    episode-start state (history filled with the phase-0 observation).
  - The clock keeps running while standing (as in training).
  - Training used init_at_random_ep_len, so any restart phase is
    in-distribution.
  - Stand mask (v15+ policies): if deploy.yaml gait_phase params contain
    stand_threshold, the phase OBSERVATION freezes at stand_phase (a
    double-stance point) whenever |cmd| < stand_threshold, while the
    underlying clock keeps counting — exactly as in training. deploy.yamls
    without stand_threshold (e.g. dist/amp_v2 / v14) are unaffected.

Usage (MuJoCo 3-terminal test):
    python3 python/h2_amp_rl_runner.py \
        --policy-onnx dist/amp_v2/policy.onnx \
        --deploy-yaml dist/amp_v2/deploy.yaml \
        --kp-scale 1.0 --warmup-time 2.0 --policy-ramp 5.0 --cmd-ramp 2.0 \
        --diag --verbose --ps4
"""

import time
import numpy as np
import argparse
import sys
import os
import yaml
import onnxruntime as ort
from collections import deque
from pathlib import Path

# Ensure we can import sibling modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from h2_shm import (
    H2SharedMemory, LowCmd,  PS4JoystickReader,
    PS2_KEY_START, PS2_KEY_SELECT, PS2_KEY_R1,
    FSM_RL_INFERENCE, FSM_DAMP,
)
# Shared machinery from the original runner (import is side-effect free)
from h2_rl_runner import (
    MotionSwitcherClient,
    joystick_to_velocity,
    ButtonDebouncer,
    get_projected_gravity,
)


# ── Gait clock ───────────────────────────────────────────────────────────────

class GaitClock:
    """Time-based biped gait phase, mirroring mdp.gait_phase_obs in training.

    stand_threshold=None disables the stand mask (v14/amp_v2 policies).
    For stand-mask policies (v15+) pass the trained threshold: obs() then
    freezes the phase at stand_phase while |cmd| < stand_threshold. The step
    counter keeps running regardless, matching training's episode clock.
    """

    def __init__(self, step_dt, gait_cycle=0.85, air_ratio=(0.38, 0.38),
                 phase_offset=(0.38, 0.88), stand_threshold=None, stand_phase=0.06):
        self.step_dt = step_dt
        self.gait_cycle = gait_cycle
        self.air_ratio = air_ratio
        self.phase_offset = phase_offset
        self.stand_threshold = stand_threshold
        self.stand_phase = stand_phase
        self.steps = 0
        self.last_t = 0.0   # effective phase used in the last obs (for diag)
        self.frozen = False

    def reset(self):
        self.steps = 0

    def advance(self):
        self.steps += 1

    def obs(self, cmd=None):
        t = (self.steps * self.step_dt / self.gait_cycle) % 1.0
        self.frozen = (
            self.stand_threshold is not None
            and cmd is not None
            and float(np.linalg.norm(cmd)) < self.stand_threshold
        )
        if self.frozen:
            t = self.stand_phase
        self.last_t = t
        left = (t + self.phase_offset[0]) % 1.0
        right = (t + self.phase_offset[1]) % 1.0
        two_pi = 2.0 * np.pi
        return np.array([
            np.sin(two_pi * left), np.cos(two_pi * left),
            np.sin(two_pi * right), np.cos(two_pi * right),
            self.air_ratio[0], self.air_ratio[1],
        ], dtype=np.float32)


# ── Observation terms ────────────────────────────────────────────────────────

def compute_obs_term(name, scale, omega, quat, cmd, q_policy, dq_policy,
                     default_pos_policy, action, gait_obs):
    """Compute a single observation term. Raises on unknown terms: silently
    substituting zeros (h2_rl_runner behaviour) breaks gait-clock policies."""
    if name == "base_ang_vel":
        return (omega * scale).astype(np.float32)
    elif name == "projected_gravity":
        return (get_projected_gravity(quat) * scale).astype(np.float32)
    elif name in ("velocity_commands", "keyboard_velocity_commands"):
        return (cmd * scale).astype(np.float32)
    elif name == "joint_pos_rel":
        return ((q_policy - default_pos_policy) * scale).astype(np.float32)
    elif name == "joint_vel_rel":
        return (dq_policy * scale).astype(np.float32)
    elif name == "last_action":
        return (action * scale).astype(np.float32)
    elif name == "gait_phase":
        return (gait_obs * scale).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown observation term '{name}' in deploy.yaml — this runner "
            f"cannot compute it, and feeding zeros would be out-of-distribution."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H2 AMP RL Inference Runner (gait-clock policies)")
    default_bundle = Path(__file__).resolve().parent.parent / "dist" / "amp_v2"
    parser.add_argument("--policy-onnx", type=str, default=str(default_bundle / "policy.onnx"),
                        help="Path to policy.onnx (default: dist/amp_v2/policy.onnx)")
    parser.add_argument("--deploy-yaml", type=str, default=str(default_bundle / "deploy.yaml"),
                        help="Path to deploy.yaml (default: dist/amp_v2/deploy.yaml)")
    parser.add_argument("--kp-scale", type=float, default=2.0,
                        help="PD gain multiplier (2.0 for MuJoCo sim, 1.0 for real robot)")
    parser.add_argument("--ankle-kp-scale", type=float, default=1.0,
                        help="Additional Kp/Kd multiplier applied to ankle joints (indices 8-11) "
                             "on top of --kp-scale. Use 0.3–0.5 if ankles oscillate on hardware. "
                             "Kd is scaled by sqrt(ankle-kp-scale) to preserve damping ratio.")
    parser.add_argument("--action-clip", type=float, default=10.0,
                        help="Clip raw policy actions to ±this value before applying action_scale. "
                             "Default 10.0 = effectively no clip. Use 0.8–1.0 on hardware to bound "
                             "ankle target excursions (with scale=0.25, clip=0.8 → ±0.2 rad from default).")
    parser.add_argument("--vx", type=float, default=0.0, help="Initial forward velocity")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial lateral velocity")
    parser.add_argument("--vyaw", type=float, default=0.0, help="Initial yaw velocity")
    parser.add_argument("--no-joystick", action="store_true",
                        help="Disable joystick input (use --vx/vy/vyaw only)")
    parser.add_argument("--auto-ai", action="store_true",
                        help="Enter AI/RL mode automatically at startup without waiting for joystick button. "
                             "Intended for headless sim testing with --no-joystick.")
    parser.add_argument("--ps4", action="store_true",
                        help="Read PS4 controller directly from Linux device instead of shared memory")
    parser.add_argument("--ps4-device", type=str, default="/dev/input/js1",
                        help="PS4 device path (default: /dev/input/js1)")
    parser.add_argument("--max-vx", type=float, default=0.5,
                        help="Max forward velocity from joystick (m/s). "
                             "v14 curriculum reached 0.8 forward; 0.5 is the safe default.")
    parser.add_argument("--max-vx-back", type=float, default=None,
                        help="Max backward velocity from joystick (m/s, positive number). "
                             "Defaults to --max-vx. Hardware testing shows sideways falls at 0.35; "
                             "use 0.25 as a safe ceiling until backward gait stability is improved.")
    parser.add_argument("--max-vy", type=float, default=0.15,
                        help="Max lateral velocity from joystick (m/s). "
                             "Lateral control is harder than forward; 0.15 is conservative.")
    parser.add_argument("--max-vyaw", type=float, default=0.8,
                        help="Max yaw rate from joystick (rad/s). v14 was trained to ±1.57; "
                             "0.8 is a conservative default, raise once sim behaviour is verified.")
    parser.add_argument("--deadzone", type=float, default=0.1,
                        help="Joystick deadzone (0.0–1.0)")
    parser.add_argument("--warmup-time", type=float, default=1.0,
                        help="Seconds to hold RL default pose before policy runs (default 1.0)")
    parser.add_argument("--policy-ramp", type=float, default=0.0,
                        help="Seconds to linearly ramp policy action authority from 0 to 1.0 after "
                             "warmup ends (default 0=instant). Use 3-5s on hardware: during the ramp "
                             "the robot holds near the default pose so the operator can safely release "
                             "the safety band before the policy is at full authority.")
    parser.add_argument("--action-ema", type=float, default=0.0,
                        help="Action EMA smoothing (default: 0=off, keep off). "
                             "WARNING: tested on H2 hardware — high alpha (0.7+) causes positive "
                             "feedback through the last_action observation and makes the robot fall "
                             "faster. The policy was trained with raw (unsmoothed) last_action; "
                             "smoothing shifts that distribution. Only use small values (0.1–0.2) "
                             "if raw actions show visible step discontinuities.")
    parser.add_argument("--cmd-ramp", type=float, default=0.0,
                        help="Velocity command ramp time in seconds (0=off, recommended: 2.0). "
                             "Limits how fast the velocity command can change: it takes --cmd-ramp "
                             "seconds to go from 0 to max-vx, and the same time to decelerate. "
                             "Prevents falls from sudden joystick speed changes on both acceleration "
                             "and deceleration. Use this instead of --cmd-filter.")
    parser.add_argument("--cmd-filter", type=float, default=0.0,
                        help="Velocity command exponential low-pass filter (0=off). Fraction of OLD "
                             "command to keep each step. Use --cmd-ramp instead.")
    parser.add_argument("--diag", action="store_true",
                        help="Print per-step diagnostics (pitch/roll, named max-error joint, "
                             "max torque, action magnitude) every 50 inference steps.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a full per-joint table (pos/target/error/vel/torque) "
                             "every 250 inference steps (~5s).")
    args = parser.parse_args()

    # ── Load Config ───────────────────────────────────────────────────────────
    deploy_yaml = Path(args.deploy_yaml)
    policy_onnx = Path(args.policy_onnx)
    if not deploy_yaml.exists():
        print(f"Error: Config not found at {deploy_yaml}")
        return
    print(f"Loading config from {deploy_yaml}")
    with open(deploy_yaml, "r") as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    step_dt = cfg["step_dt"]
    # joint_ids_map: 12-element list mapping policy_idx -> hardware motor index.
    joint_ids_map = cfg["joint_ids_map"]
    default_pos_policy = np.array(cfg["default_joint_pos"], dtype=np.float32)
    num_joints = len(default_pos_policy)
    if len(joint_ids_map) != num_joints:
        print(f"Error: joint_ids_map has {len(joint_ids_map)} entries but "
              f"default_joint_pos has {num_joints} — this deploy.yaml is not in "
              f"the curated 12-joint policy-order format (see dist/amp_v2).")
        return

    kp_scale = args.kp_scale
    kd_scale = np.sqrt(kp_scale)  # heuristic

    kps = np.array(cfg["stiffness"], dtype=np.float32) * kp_scale
    kds = np.array(cfg["damping"], dtype=np.float32) * kd_scale

    # Per-joint ankle gain override (indices 8-11: L_ank_p, R_ank_p, L_ank_r, R_ank_r)
    ankle_kp_scale = args.ankle_kp_scale
    if ankle_kp_scale != 1.0:
        ankle_kd_scale = np.sqrt(ankle_kp_scale)
        kps[8:12] *= ankle_kp_scale
        kds[8:12] *= ankle_kd_scale

    # Action scaling
    action_cfg = cfg["actions"]["JointPositionAction"]
    action_scale = np.array(action_cfg["scale"], dtype=np.float32)
    action_offset = np.array(action_cfg["offset"], dtype=np.float32)
    action_clip = args.action_clip

    ankle_str = f"  ankle×{ankle_kp_scale:.2f}" if ankle_kp_scale != 1.0 else ""
    print(f"Joint gains (kp_scale={kp_scale:.1f}{ankle_str}):")
    joint_names = ["L_hip_p","R_hip_p","L_hip_r","R_hip_r",
                   "L_hip_y","R_hip_y","L_knee","R_knee",
                   "L_ank_p","R_ank_p","L_ank_r","R_ank_r"]
    for i, (name, kp, kd) in enumerate(zip(joint_names[:num_joints], kps, kds)):
        print(f"  [{i:2d}] {name:10s}  kp={kp:6.1f}  kd={kd:.2f}")

    # Observation setup
    obs_cfg = cfg["observations"]
    obs_groups = []
    for name, ocfg in obs_cfg.items():
        scale = np.array(ocfg["scale"], dtype=np.float32)
        history = ocfg.get("history_length", 1)
        obs_groups.append({"name": name, "scale": scale, "dim": len(scale),
                           "history": history, "params": ocfg.get("params") or {}})

    # Gait clock, parameterized from the deploy.yaml gait_phase entry
    gait_clock = None
    for g in obs_groups:
        if g["name"] == "gait_phase":
            p = g["params"]
            gait_clock = GaitClock(
                step_dt,
                gait_cycle=p.get("gait_cycle", 0.85),
                air_ratio=tuple(p.get("air_ratio", (0.38, 0.38))),
                phase_offset=tuple(p.get("phase_offset", (0.38, 0.88))),
                stand_threshold=p.get("stand_threshold"),
                stand_phase=p.get("stand_phase", 0.06),
            )
            if gait_clock.stand_threshold is not None:
                stand_str = (f"  stand-mask: freeze at t={gait_clock.stand_phase} "
                             f"when |cmd|<{gait_clock.stand_threshold}")
            else:
                stand_str = "  stand-mask: off (pre-v15 policy)"
            print(f"Gait clock: cycle={gait_clock.gait_cycle}s "
                  f"offsets={gait_clock.phase_offset} air_ratio={gait_clock.air_ratio}"
                  f"{stand_str}")
    if gait_clock is None:
        print("NOTE: no gait_phase obs in deploy.yaml — plain velocity policy; "
              "h2_rl_runner.py would work equally well.")
        gait_clock = GaitClock(step_dt)  # unused, keeps obs code uniform

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"Loading ONNX model: {policy_onnx}")
    session = ort.InferenceSession(str(policy_onnx))
    input_name = session.get_inputs()[0].name

    onnx_obs_dim = session.get_inputs()[0].shape[-1]
    single_obs_dim = sum(g["dim"] for g in obs_groups)
    history_len = obs_groups[0]["history"]

    expected_combined = single_obs_dim * history_len
    if onnx_obs_dim != expected_combined:
        print(f"Adapting history length: ONNX expects {onnx_obs_dim}, calc {expected_combined}")
        if onnx_obs_dim % single_obs_dim == 0:
            history_len = onnx_obs_dim // single_obs_dim
            for g in obs_groups:
                g["history"] = history_len
        else:
            print(f"Error: ONNX obs dim {onnx_obs_dim} is not a multiple of the "
                  f"single-frame obs dim {single_obs_dim} — deploy.yaml does not "
                  f"match this policy. Refusing to run with a padded observation.")
            return

    # ── Initialize Shared Memory ──────────────────────────────────────────────
    try:
        h2 = H2SharedMemory()
    except Exception as e:
        print(f"Failed to connect to shared memory: {e}")
        return

    print("Connected to Shared Memory.")

    # ── Initialize PS4 Controller (if requested) ──────────────────────────────
    ps4_reader = None
    if not args.no_joystick and args.ps4:
        try:
            ps4_reader = PS4JoystickReader(device=args.ps4_device)
        except Exception as e:
            print(f"[PS4] Failed to open controller: {e}")
            print("[PS4] Falling back to shared memory joystick")

    # ── Initialize DDS Client ─────────────────────────────────────────────────
    dds_client = MotionSwitcherClient()

    # ── State Variables ───────────────────────────────────────────────────────
    action = np.zeros(num_joints, dtype=np.float32)
    smoothed_action = np.zeros(num_joints, dtype=np.float32)
    motor_fault_prev = False  # edge-detect: log only on fault transitions
    cmd_target = np.array([args.vx, args.vy, args.vyaw], dtype=np.float32)  # raw joystick command
    cmd = np.array([args.vx, args.vy, args.vyaw], dtype=np.float32)         # filtered command (sent to policy)
    max_vx_back = args.max_vx_back if args.max_vx_back is not None else None
    if args.cmd_ramp > 0.0:
        cmd_max_rates = np.array(
            [args.max_vx, args.max_vy, args.max_vyaw], dtype=np.float32
        ) * (step_dt / args.cmd_ramp)
    else:
        cmd_max_rates = None
    use_joystick = not args.no_joystick
    debouncer = ButtonDebouncer(cooldown_sec=0.5)
    ai_mode_active = False
    auto_ai_triggered = False
    inference_count = 0
    WARMUP_STEPS = max(1, round(args.warmup_time / step_dt))
    warmup_remaining = 0
    POLICY_RAMP_STEPS = max(0, round(args.policy_ramp / step_dt))
    policy_ramp_step = 0

    # Initialize histories from one state read
    state = h2.get_low_state()
    quat = np.array(state.imu_state.quaternion)
    omega = np.array(state.imu_state.gyroscope)
    q_all = np.array([m.q for m in state.motor_state])
    dq_all = np.array([m.dq for m in state.motor_state])
    q_policy = np.array([q_all[joint_ids_map[i]] for i in range(num_joints)], dtype=np.float32)
    dq_policy = np.array([dq_all[joint_ids_map[i]] for i in range(num_joints)], dtype=np.float32)

    group_histories = []
    gait_obs = gait_clock.obs(cmd)
    for g in obs_groups:
        init_val = compute_obs_term(
            g["name"], g["scale"], omega, quat, cmd,
            q_policy, dq_policy, default_pos_policy, action, gait_obs
        )
        hist = deque(maxlen=history_len)
        for _ in range(history_len):
            hist.append(init_val.copy())
        group_histories.append(hist)

    low_cmd = LowCmd()
    # Gains in deploy.yaml are in POLICY order — must remap to hardware indices.
    for policy_idx in range(num_joints):
        hw_idx = joint_ids_map[policy_idx]
        low_cmd.motor_cmd[hw_idx].kp = float(kps[policy_idx])
        low_cmd.motor_cmd[hw_idx].kd = float(kds[policy_idx])

    print("=" * 60)
    if use_joystick:
        if ps4_reader is not None:
            print("PS4 Controller Direct Mode")
            print(f"  Device      : {args.ps4_device}")
        else:
            print("Joystick Control Enabled (via shared memory)")
        print("  Left  Stick : forward/backward (vx), strafe (vy)")
        print("  Right Stick : yaw rotation (vyaw)")
        print("  OPTIONS/START  : switch to AI mode (RL inference)")
        print("  SHARE/SELECT   : switch back to normal (DAMP)")
        print("  R1             : emergency DAMP")
        back_limit = max_vx_back if max_vx_back is not None else args.max_vx
        print(f"  Max velocity: fwd={args.max_vx} back={back_limit} vy={args.max_vy} vyaw={args.max_vyaw}")
        print(f"  Deadzone    : {args.deadzone}")
    else:
        print(f"Joystick disabled. Fixed cmd: vx={args.vx} vy={args.vy} vyaw={args.vyaw}")
    if cmd_max_rates is not None:
        print(f"  Cmd ramp    : {args.cmd_ramp:.1f}s (0→max in {args.cmd_ramp:.1f}s, same for stop)")
    elif args.cmd_filter > 0.0:
        print(f"  Cmd filter  : alpha={args.cmd_filter} (exponential)")
    if args.action_ema > 0.0:
        print(f"  Action EMA  : alpha={args.action_ema} ({args.action_ema*100:.0f}% old each step)")
    if action_clip < 10.0:
        print(f"  Action clip : ±{action_clip:.2f}  (→ ±{action_clip*0.25:.3f} rad from default with scale=0.25)")
    if POLICY_RAMP_STEPS > 0:
        print(f"  Policy ramp : {args.policy_ramp:.1f}s after warmup ({POLICY_RAMP_STEPS} steps, 0→full authority)")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    def enter_ai_mode():
        nonlocal ai_mode_active, warmup_remaining, policy_ramp_step, action, smoothed_action
        dds_client.select_mode("ai")           # notify MotionSwitcherServer (best-effort)
        h2.set_fsm_id(FSM_RL_INFERENCE)        # always write SHM directly — reliable
        ai_mode_active = True
        warmup_remaining = WARMUP_STEPS
        policy_ramp_step = 0
        action = np.zeros(num_joints, dtype=np.float32)
        smoothed_action = np.zeros(num_joints, dtype=np.float32)
        # Snap cmd to current target on mode entry to avoid stale ramp state
        cmd[:] = cmd_target
        # Gait clock restarts at phase 0, matching Isaac Lab episode start
        gait_clock.reset()
        print(f"[RL] Warmup: holding defaults for {args.warmup_time:.1f}s ({WARMUP_STEPS} steps)")

    try:
        while True:
            cycle_start = time.time()

            # ── 0. Auto-AI entry (headless sim mode) ──────────────────────
            if args.auto_ai and not auto_ai_triggered and not ai_mode_active:
                print("[AUTO] --auto-ai: entering RL mode automatically")
                enter_ai_mode()
                auto_ai_triggered = True

            # ── 1. Read Joystick & Handle Buttons ─────────────────────────
            if use_joystick:
                if ps4_reader is not None:
                    js = ps4_reader.read()
                else:
                    js = h2.get_joystick_state()

                if js.valid:
                    vx, vy, vyaw = joystick_to_velocity(
                        js, args.max_vx, args.max_vy, args.max_vyaw, args.deadzone)
                    if max_vx_back is not None and vx < 0:
                        vx = max(vx, -max_vx_back)
                    cmd_target[0] = vx
                    cmd_target[1] = vy
                    cmd_target[2] = vyaw

                    # OPTIONS/START → enter AI mode
                    if debouncer.pressed(js.keys, PS2_KEY_START):
                        if not ai_mode_active:
                            print("[JOY] OPTIONS pressed → switching to AI mode")
                            enter_ai_mode()

                    # SHARE/SELECT → exit AI mode
                    if debouncer.pressed(js.keys, PS2_KEY_SELECT):
                        if ai_mode_active:
                            print("[JOY] SHARE pressed → exiting AI mode")
                            dds_client.release_mode()
                            h2.set_fsm_id(FSM_DAMP)
                            ai_mode_active = False

                    # R1 → emergency stop (DAMP)
                    if debouncer.pressed(js.keys, PS2_KEY_R1):
                        print("[JOY] R1 pressed → emergency DAMP")
                        h2.set_fsm_id(FSM_DAMP)
                        ai_mode_active = False

            # ── 2. Check FSM state ───────────────────────────────────────
            current_fsm = h2.get_fsm_id()
            if ai_mode_active and current_fsm != FSM_RL_INFERENCE:
                # FSM left RL_INFERENCE (e.g. watchdog or safety trigger)
                ai_mode_active = False
                warmup_remaining = 0
                policy_ramp_step = 0

            # ── 3. Read Sensor State ─────────────────────────────────────
            state = h2.get_low_state()

            quat = np.array(state.imu_state.quaternion)
            omega = np.array(state.imu_state.gyroscope)
            rpy = np.array(state.imu_state.rpy)  # [roll, pitch, yaw] in radians

            q_all   = np.array([m.q       for m in state.motor_state])
            dq_all  = np.array([m.dq      for m in state.motor_state])
            tau_all = np.array([m.tau_est for m in state.motor_state])

            q_policy   = np.array([q_all[joint_ids_map[i]]   for i in range(num_joints)], dtype=np.float32)
            dq_policy  = np.array([dq_all[joint_ids_map[i]]  for i in range(num_joints)], dtype=np.float32)
            tau_policy = np.array([tau_all[joint_ids_map[i]] for i in range(num_joints)], dtype=np.float32)

            # ── 3b. Motor Fault Check ────────────────────────────────────
            fault_active = state.motor_fault_active
            if fault_active and not motor_fault_prev:
                fid = state.faulted_motor_id
                print(f"[FAULT] Motor {fid} fault — FSM forced to DAMP, stopping inference")
                if ai_mode_active:
                    ai_mode_active = False
                    warmup_remaining = 0
                    h2.set_fsm_id(FSM_DAMP)
            elif not fault_active and motor_fault_prev:
                print("[FAULT] Motor fault cleared")
            motor_fault_prev = fault_active

            # ── 3d. Command Smoothing ────────────────────────────────────
            if cmd_max_rates is not None:
                delta = cmd_target - cmd
                cmd[:] = cmd + np.clip(delta, -cmd_max_rates, cmd_max_rates)
            elif args.cmd_filter > 0.0:
                cmd[:] = args.cmd_filter * cmd + (1.0 - args.cmd_filter) * cmd_target
            else:
                cmd[:] = cmd_target

            # ── 4. Update Observation History ────────────────────────────
            # Gait clock: phase 0 through warmup (matches episode-start history),
            # then advances one step per control cycle while the policy runs.
            in_warmup = current_fsm == FSM_RL_INFERENCE and warmup_remaining > 0
            if current_fsm == FSM_RL_INFERENCE and not in_warmup and ai_mode_active:
                gait_clock.advance()
            # Pass the ramped cmd (what the policy sees) so the stand-mask
            # freeze engages/disengages exactly as it did in training.
            gait_obs = gait_clock.obs(cmd)

            for gi, g in enumerate(obs_groups):
                val = compute_obs_term(
                    g["name"], g["scale"], omega, quat, cmd,
                    q_policy, dq_policy, default_pos_policy, action, gait_obs
                )
                group_histories[gi].append(val)

            # ── 5. Assemble Full Observation ─────────────────────────────
            obs_parts = []
            for hist in group_histories:
                obs_parts.extend(hist)
            full_obs = np.concatenate(obs_parts)

            # ── 6. Inference ─────────────────────────────────────────────
            # Skip inference during warmup: keeping action=0 throughout warmup
            # ensures the last_action obs history is all-zeros at policy step 1,
            # matching Isaac Lab's episode-start initialization.
            if not in_warmup:
                obs_input = full_obs.reshape(1, -1).astype(np.float32)
                onnx_output = session.run(None, {input_name: obs_input})
                raw_action = onnx_output[0].flatten()[:num_joints]
                raw_action = np.clip(raw_action, -action_clip, action_clip)

                if args.action_ema > 0.0:
                    smoothed_action = (args.action_ema * smoothed_action
                                       + (1.0 - args.action_ema) * raw_action)
                    action = smoothed_action
                else:
                    action = raw_action
            # else: action stays at 0 (set at AI mode entry) — last_action obs = 0

            # ── 7. Send Commands (only when FSM is in RL_INFERENCE) ──────
            if current_fsm == FSM_RL_INFERENCE:
                if warmup_remaining > 0:
                    # Warmup: hold RL default positions (action=0 → offset).
                    target_pos_warmup = action_offset
                    for policy_idx in range(num_joints):
                        hw_idx = joint_ids_map[policy_idx]
                        low_cmd.motor_cmd[hw_idx].q = float(target_pos_warmup[policy_idx])
                        low_cmd.motor_cmd[hw_idx].dq = 0.0
                        low_cmd.motor_cmd[hw_idx].tau = 0.0
                    h2.set_low_cmd(low_cmd)
                    warmup_remaining -= 1
                    if warmup_remaining == 0:
                        if POLICY_RAMP_STEPS > 0:
                            print(f"[RL] Warmup complete — ramping policy authority "
                                  f"over {args.policy_ramp:.1f}s. Release band now.")
                        else:
                            print("[RL] Warmup complete — policy control active")
                else:
                    if POLICY_RAMP_STEPS > 0 and policy_ramp_step < POLICY_RAMP_STEPS:
                        ramp_alpha = policy_ramp_step / POLICY_RAMP_STEPS
                        policy_ramp_step += 1
                    else:
                        ramp_alpha = 1.0
                    target_pos_policy = ramp_alpha * action * action_scale + action_offset
                    for policy_idx in range(num_joints):
                        hw_idx = joint_ids_map[policy_idx]
                        low_cmd.motor_cmd[hw_idx].q = float(target_pos_policy[policy_idx])
                        low_cmd.motor_cmd[hw_idx].dq = 0.0
                        low_cmd.motor_cmd[hw_idx].tau = 0.0
                    h2.set_low_cmd(low_cmd)
                    inference_count += 1

                    # ── Diagnostic logging ────────────────────────────────
                    if args.diag and inference_count % 50 == 1:
                        q_err = q_policy - default_pos_policy
                        q_err_abs = np.abs(q_err)
                        max_err_idx = int(np.argmax(q_err_abs))
                        pitch_deg = float(np.degrees(rpy[1]))
                        roll_deg  = float(np.degrees(rpy[0]))
                        tau_max   = float(np.max(np.abs(tau_policy)))
                        phase_str = f"{gait_clock.last_t:.2f}" + ("*" if gait_clock.frozen else "")
                        print(f"[DIAG] #{inference_count:5d}  "
                              f"pitch={pitch_deg:+.1f}° roll={roll_deg:+.1f}°  "
                              f"q_err_max={q_err_abs[max_err_idx]:.3f}[{joint_names[max_err_idx]}]  "
                              f"tau_max={tau_max:.1f}  "
                              f"act_max={np.max(np.abs(action)):.3f}  "
                              f"phase={phase_str}  "
                              f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})")

                    # ── Verbose joint table ───────────────────────────────
                    if args.verbose and inference_count % 250 == 1:
                        target_pos_disp = action * action_scale + action_offset
                        q_err = q_policy - default_pos_policy
                        t_sec = inference_count * step_dt
                        print(f"[JOINT] #{inference_count:5d}  t={t_sec:.1f}s  "
                              f"pitch={float(np.degrees(rpy[1])):+.1f}°  "
                              f"roll={float(np.degrees(rpy[0])):+.1f}°")
                        print(f"  {'Joint':<10}  {'q':>7}  {'target':>7}  {'err':>7}  "
                              f"{'dq':>6}  {'tau':>6}")
                        for i in range(num_joints):
                            print(f"  {joint_names[i]:<10}  "
                                  f"{q_policy[i]:+7.3f}  "
                                  f"{target_pos_disp[i]:+7.3f}  "
                                  f"{q_err[i]:+7.3f}  "
                                  f"{dq_policy[i]:+6.2f}  "
                                  f"{tau_policy[i]:+6.1f}")

            # ── 8. Periodic Status Log ───────────────────────────────────
            if inference_count % 500 == 1 and current_fsm == FSM_RL_INFERENCE:
                print(f"[RL] infer #{inference_count}  "
                      f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})  "
                      f"fsm={current_fsm}")

            # ── 9. Sleep ─────────────────────────────────────────────────
            elapsed = time.time() - cycle_start
            sleep = step_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopping...")
        if ai_mode_active:
            dds_client.release_mode()
            h2.set_fsm_id(FSM_DAMP)
            print("Released AI mode on shutdown")
    finally:
        if ps4_reader is not None:
            ps4_reader.close()
        h2.close()
        print("Done.")


if __name__ == "__main__":
    main()
