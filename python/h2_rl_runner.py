
import time
import json
import numpy as np
import argparse
import sys
import os
import yaml
import onnxruntime as ort
from collections import deque
from pathlib import Path

# Ensure we can import h2_shm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from h2_shm import (
    H2SharedMemory, LowCmd, PS4JoystickReader,
    PS2_KEY_START, PS2_KEY_SELECT, PS2_KEY_R1, PS2_KEY_L1,
    FSM_RL_INFERENCE, FSM_DAMP,
    MOTION_SWITCHER_API_ID_SELECT_MODE, MOTION_SWITCHER_API_ID_RELEASE_MODE,
)

# ── DDS Service Client ──────────────────────────────────────────────────────

class MotionSwitcherClient:
    """DDS client for calling MotionSwitcher service to switch FSM modes.

    Uses cyclonedds Python bindings to send Request messages to the
    motion_switcher DDS service (same protocol as topstar_sdk2).
    Falls back to direct shared-memory writes if cyclonedds is unavailable.
    """

    def __init__(self):
        self.available = False
        self._req_id = int(time.time()) & 0x7FFFFFFF  # unique starting id
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.pub import DataWriter
            from cyclonedds.core import Qos, Policy
            from cyclonedds.idl import IdlStruct
            from cyclonedds.idl.types import int32, int64, uint8, sequence
            from dataclasses import dataclass

            # ── Define DDS types matching topstar_api.idl ────────────────
            @dataclass
            class RequestIdentity(IdlStruct,
                                  typename="topstar_api::msg::dds_::RequestIdentity_"):
                id: int64
                api_id: int32

            @dataclass
            class RequestLease(IdlStruct,
                               typename="topstar_api::msg::dds_::RequestLease_"):
                id: int64

            @dataclass
            class RequestPolicy(IdlStruct,
                                typename="topstar_api::msg::dds_::RequestPolicy_"):
                priority: int32
                noreply: bool

            @dataclass
            class RequestHeader(IdlStruct,
                                typename="topstar_api::msg::dds_::RequestHeader_"):
                identity: RequestIdentity
                lease: RequestLease
                policy: RequestPolicy

            @dataclass
            class Request(IdlStruct,
                          typename="topstar_api::msg::dds_::Request_"):
                header: RequestHeader
                parameter: str
                binary: sequence[uint8]

            self._types = {
                "Request": Request,
                "RequestHeader": RequestHeader,
                "RequestIdentity": RequestIdentity,
                "RequestLease": RequestLease,
                "RequestPolicy": RequestPolicy,
            }

            # ── Create DDS entities ──────────────────────────────────────
            participant = DomainParticipant(0)
            qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=1_000_000_000),
                Policy.Durability.Volatile,
                Policy.History.KeepLast(10),
            )

            ms_topic = Topic(participant,
                             "rt/api/motion_switcher/request",
                             Request, qos=qos)
            self._ms_writer = DataWriter(participant, ms_topic, qos=qos)

            self.available = True
            print("[DDS] MotionSwitcher client ready")

        except ImportError:
            print("[DDS] cyclonedds not installed – using shared memory fallback")
        except Exception as e:
            print(f"[DDS] Init failed ({e}) – using shared memory fallback")

    def _next_id(self):
        self._req_id += 1
        return self._req_id

    def _make_request(self, api_id, parameter="{}"):
        T = self._types
        return T["Request"](
            header=T["RequestHeader"](
                identity=T["RequestIdentity"](id=self._next_id(), api_id=api_id),
                lease=T["RequestLease"](id=0),
                policy=T["RequestPolicy"](priority=0, noreply=False),
            ),
            parameter=parameter,
            binary=[],
        )

    def select_mode(self, mode_name):
        """Send select_mode request (api_id 1002) to MotionSwitcher."""
        if not self.available:
            return False
        try:
            param = json.dumps({"name": mode_name})
            self._ms_writer.write(
                self._make_request(MOTION_SWITCHER_API_ID_SELECT_MODE, param))
            print(f"[DDS] select_mode → {mode_name}")
            return True
        except Exception as e:
            print(f"[DDS] select_mode failed: {e}")
            return False

    def release_mode(self):
        """Send release_mode request (api_id 1003) to MotionSwitcher."""
        if not self.available:
            return False
        try:
            param = json.dumps({"name": "ai"})
            self._ms_writer.write(
                self._make_request(MOTION_SWITCHER_API_ID_RELEASE_MODE, param))
            print("[DDS] release_mode → normal")
            return True
        except Exception as e:
            print(f"[DDS] release_mode failed: {e}")
            return False


# ── Joystick Helpers ─────────────────────────────────────────────────────────

def apply_deadzone(value, deadzone):
    """Remove stick drift around center and rescale remaining range to 0–1."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def joystick_to_velocity(js, max_vx, max_vy, max_vyaw, deadzone):
    """Map PS2 stick axes to robot velocity command.

    Linux joystick convention: up/left = negative.
    Robot convention: vx>0 forward, vy>0 left, vyaw>0 CCW.
    """
    vx   = -apply_deadzone(js.ly, deadzone) * max_vx
    vy   = -apply_deadzone(js.lx, deadzone) * max_vy
    vyaw = -apply_deadzone(js.rx, deadzone) * max_vyaw
    return vx, vy, vyaw


class ButtonDebouncer:
    """Simple cooldown-based debouncer for joystick buttons."""

    def __init__(self, cooldown_sec=0.5):
        self.cooldown = cooldown_sec
        self._last = {}

    def pressed(self, keys, mask):
        """Return True once per press (cooldown-gated)."""
        if not (keys & mask):
            return False
        now = time.time()
        if (now - self._last.get(mask, 0)) >= self.cooldown:
            self._last[mask] = now
            return True
        return False


# ── Policy Helpers (unchanged) ───────────────────────────────────────────────

def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse (conjugate) of a quaternion.
    quat is (w, x, y, z) convention."""
    w, x, y, z = quat
    q_vec = np.array([-x, -y, -z])
    t = 2.0 * np.cross(q_vec, vec)
    return vec + w * t + np.cross(q_vec, t)

def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Project world gravity [0,0,-1] into body frame."""
    gravity_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_world).astype(np.float32)

def compute_obs_term(name, scale, omega, quat, cmd, q_policy, dq_policy,
                     default_pos_policy, action, dim):
    """Compute a single observation term."""
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
    else:
        return np.zeros(dim, dtype=np.float32)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H2 RL Inference Runner")
    parser.add_argument("--lab-path", type=str, default="/home/test/topstar_rl_lab",
                        help="Path to topstar_rl_lab root")
    parser.add_argument("--policy-onnx", type=str, default=None,
                        help="Direct path to policy.onnx (overrides --lab-path lookup)")
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Direct path to deploy.yaml (overrides --lab-path lookup)")
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
                             "Training curriculum was ±0.5 m/s; stay at or below 0.5 for robustness.")
    parser.add_argument("--max-vx-back", type=float, default=None,
                        help="Max backward velocity from joystick (m/s, positive number). "
                             "Defaults to --max-vx. Hardware testing shows sideways falls at 0.35; "
                             "use 0.25 as a safe ceiling until backward gait stability is improved.")
    parser.add_argument("--max-vy", type=float, default=0.15,
                        help="Max lateral velocity from joystick (m/s). "
                             "Lateral control is harder than forward; 0.15 is conservative.")
    parser.add_argument("--max-vyaw", type=float, default=0.8,
                        help="Max yaw rate from joystick (rad/s)")
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
                             "command to keep each step. alpha=0.85 ramps in ~0.13s, alpha=0.95 "
                             "in ~0.4s. Use --cmd-ramp instead for more predictable behaviour.")
    parser.add_argument("--diag", action="store_true",
                        help="Print per-step diagnostics (pitch/roll, named max-error joint, "
                             "max torque, action magnitude) every 50 inference steps.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a full per-joint table (pos/target/error/vel/torque) "
                             "every 250 inference steps (~5s). Use on hardware to identify "
                             "which joint is struggling or saturating.")
    args = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────
    lab_path = Path(args.lab_path)
    policy_dir = lab_path / "deploy" / "robots" / "h2" / "config" / "policy" / "velocity" / "v0"
    deploy_yaml = Path(args.deploy_yaml) if args.deploy_yaml else policy_dir / "params" / "deploy.yaml"
    policy_onnx = Path(args.policy_onnx) if args.policy_onnx else policy_dir / "exported" / "policy.onnx"

    if not deploy_yaml.exists():
        print(f"Error: Config not found at {deploy_yaml}")
        return

    # ── Load Config ───────────────────────────────────────────────────────────
    print(f"Loading config from {deploy_yaml}")
    with open(deploy_yaml, "r") as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    step_dt = cfg["step_dt"]
    # joint_ids_map: 12-element list mapping policy_idx -> hardware motor index.
    # deploy.yaml comment documents the mapping (e.g. policy[0]=LHipPitch -> hw 0).
    joint_ids_map = cfg["joint_ids_map"]
    default_pos_policy = np.array(cfg["default_joint_pos"], dtype=np.float32)
    num_joints = len(default_pos_policy)

    # Note: On real robot, the motor driver executes PD.
    # The policy outputs target positions.
    # We use the config stiffness/damping scaled by args.kp_scale if needed.
    kp_scale = args.kp_scale
    kd_scale = np.sqrt(kp_scale) # heuristic

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
        obs_groups.append({"name": name, "scale": scale, "dim": len(scale), "history": history})

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"Loading ONNX model: {policy_onnx}")
    session = ort.InferenceSession(str(policy_onnx))
    input_name = session.get_inputs()[0].name

    # Verify input shape
    onnx_obs_dim = session.get_inputs()[0].shape[-1]
    single_obs_dim = sum(g["dim"] for g in obs_groups)
    history_len = obs_groups[0]["history"]

    # Fix history length mismatch if any (adaptation from original script)
    expected_combined = single_obs_dim * history_len
    if onnx_obs_dim != expected_combined:
        print(f"Adapting history length: ONNX expects {onnx_obs_dim}, calc {expected_combined}")
        if onnx_obs_dim % single_obs_dim == 0:
            history_len = onnx_obs_dim // single_obs_dim
            for g in obs_groups:
                g['history'] = history_len

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
    # Resolve backward velocity cap (None → same as forward limit)
    max_vx_back = args.max_vx_back if args.max_vx_back is not None else None
    # Pre-compute max rate of change per step for the linear rate limiter.
    # cmd_ramp seconds to go from 0 → max; same rate applied to deceleration.
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
    # Warmup: number of steps to hold RL default positions before policy runs.
    # STAND_UP uses a deeper crouch than the RL training defaults, so we need
    # time for joints to transition (esp. elbows: 0 → 0.97 rad, knee: 0.6 → 0.3).
    WARMUP_STEPS = max(1, round(args.warmup_time / step_dt))
    warmup_remaining = 0
    POLICY_RAMP_STEPS = max(0, round(args.policy_ramp / step_dt))
    policy_ramp_step = 0  # counts up from 0 to POLICY_RAMP_STEPS after warmup ends

    # Initialize histories
    group_histories = []
    # Need initial observation to fill history
    # We'll read one state to init
    state = h2.get_low_state()

    quat = np.array(state.imu_state.quaternion)
    omega = np.array(state.imu_state.gyroscope)

    # Map raw motor state to policy order
    q_all = np.array([m.q for m in state.motor_state])
    dq_all = np.array([m.dq for m in state.motor_state])

    q_policy = np.array([q_all[joint_ids_map[i]] for i in range(num_joints)], dtype=np.float32)
    dq_policy = np.array([dq_all[joint_ids_map[i]] for i in range(num_joints)], dtype=np.float32)

    for g in obs_groups:
        init_val = compute_obs_term(
            g["name"], g["scale"], omega, quat, cmd,
            q_policy, dq_policy, default_pos_policy, action, g["dim"]
        )
        hist = deque(maxlen=history_len)
        for _ in range(history_len):
            hist.append(init_val.copy())
        group_histories.append(hist)

    # Command object for reuse
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
            print("  Left  Stick : forward/backward (vx), strafe (vy)")
            print("  Right Stick : yaw rotation (vyaw)")
            print("  OPTIONS     : switch to AI mode (RL inference)")
            print("  SHARE       : switch back to normal (DAMP)")
            print("  R1          : emergency DAMP")
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

    try:
        while True:
            cycle_start = time.time()

            # ── 0. Auto-AI entry (headless sim mode) ──────────────────────
            if args.auto_ai and not auto_ai_triggered and not ai_mode_active:
                print("[AUTO] --auto-ai: entering RL mode automatically")
                dds_client.select_mode("ai")
                h2.set_fsm_id(FSM_RL_INFERENCE)
                ai_mode_active = True
                auto_ai_triggered = True
                warmup_remaining = WARMUP_STEPS
                policy_ramp_step = 0
                action = np.zeros(num_joints, dtype=np.float32)
                smoothed_action = np.zeros(num_joints, dtype=np.float32)
                cmd[:] = cmd_target
                print(f"[RL] Warmup: holding defaults for {args.warmup_time:.1f}s ({WARMUP_STEPS} steps)")

            # ── 1. Read Joystick & Handle Buttons ─────────────────────────
            if use_joystick:
                if ps4_reader is not None:
                    js = ps4_reader.read()
                else:
                    js = h2.get_joystick_state()

                if js.valid:
                    # Map sticks to velocity commands (written to cmd_target; cmd is filtered below)
                    vx, vy, vyaw = joystick_to_velocity(
                        js, args.max_vx, args.max_vy, args.max_vyaw, args.deadzone)
                    # Optionally cap backward speed lower than forward (backward is less stable)
                    if max_vx_back is not None and vx < 0:
                        vx = max(vx, -max_vx_back)
                    cmd_target[0] = vx
                    cmd_target[1] = vy
                    cmd_target[2] = vyaw

                    # OPTIONS/START → enter AI mode
                    if debouncer.pressed(js.keys, PS2_KEY_START):
                        if not ai_mode_active:
                            print("[JOY] OPTIONS pressed → switching to AI mode")
                            dds_client.select_mode("ai")  # notify MotionSwitcherServer (best-effort)
                            h2.set_fsm_id(FSM_RL_INFERENCE)  # always write SHM directly — reliable
                            ai_mode_active = True
                            warmup_remaining = WARMUP_STEPS
                            policy_ramp_step = 0
                            action = np.zeros(num_joints, dtype=np.float32)
                            smoothed_action = np.zeros(num_joints, dtype=np.float32)
                            # Snap cmd to current target on mode entry to avoid stale ramp state
                            cmd[:] = cmd_target
                            print(f"[RL] Warmup: holding defaults for "
                                  f"{args.warmup_time:.1f}s ({WARMUP_STEPS} steps)")

                    # SHARE/SELECT → exit AI mode
                    if debouncer.pressed(js.keys, PS2_KEY_SELECT):
                        if ai_mode_active:
                            print("[JOY] SHARE pressed → exiting AI mode")
                            dds_client.release_mode()  # notify MotionSwitcherServer (best-effort)
                            h2.set_fsm_id(FSM_DAMP)   # always write SHM directly — reliable
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

            # Map motor state
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
            # Smooth the velocity command seen by the policy to prevent falls
            # from sudden joystick changes (both acceleration AND deceleration).
            if cmd_max_rates is not None:
                # Linear rate limiter (--cmd-ramp): clip Δv per step.
                # Symmetric: same ramp rate for speed-up and slow-down.
                delta = cmd_target - cmd
                cmd[:] = cmd + np.clip(delta, -cmd_max_rates, cmd_max_rates)
            elif args.cmd_filter > 0.0:
                # Exponential low-pass filter (--cmd-filter, legacy).
                cmd[:] = args.cmd_filter * cmd + (1.0 - args.cmd_filter) * cmd_target
            else:
                cmd[:] = cmd_target

            # ── 4. Update Observation History ────────────────────────────
            for gi, g in enumerate(obs_groups):
                val = compute_obs_term(
                    g["name"], g["scale"], omega, quat, cmd,
                    q_policy, dq_policy, default_pos_policy, action, g["dim"]
                )
                group_histories[gi].append(val)

            # ── 5. Assemble Full Observation ─────────────────────────────
            obs_parts = []
            for hist in group_histories:
                obs_parts.extend(hist)
            full_obs = np.concatenate(obs_parts)

            # Pad/Truncate if needed
            if len(full_obs) != onnx_obs_dim:
                if len(full_obs) < onnx_obs_dim:
                    full_obs = np.pad(full_obs, (0, onnx_obs_dim - len(full_obs)))
                else:
                    full_obs = full_obs[:onnx_obs_dim]

            # ── 6. Inference ─────────────────────────────────────────────
            # Skip inference during warmup: keeping action=0 throughout warmup
            # ensures the last_action obs history is all-zeros at policy step 1,
            # matching Isaac Lab's episode-start initialization (CircularBuffer
            # reset to zero, first append fills all slots with action=0).
            # Running inference during warmup and storing its output in `action`
            # pollutes the last_action history with non-zero values and causes
            # the policy to output large corrective actions from step 1 onward.
            in_warmup = current_fsm == FSM_RL_INFERENCE and warmup_remaining > 0
            if not in_warmup:
                obs_input = full_obs.reshape(1, -1).astype(np.float32)
                onnx_output = session.run(None, {input_name: obs_input})
                raw_action = onnx_output[0].flatten()[:num_joints]
                raw_action = np.clip(raw_action, -action_clip, action_clip)

                # Action EMA smoothing: reduces jitter from sim-to-real gap.
                # The smoothed action is used both as the motor command AND as the
                # last_action observation for the next step, keeping them consistent.
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
                    # Warmup phase: hold all joints at RL default positions.
                    # action=0 → target = 0*scale + offset = offset = defaults.
                    # This bridges the gap between the FSM STAND_UP pose
                    # (deeper crouch + passive arms) and the RL training defaults
                    # (shallower stand, elbows at 0.97 rad, etc.) before the
                    # policy output would see large initial errors.
                    target_pos_warmup = action_offset  # action=0 → just the offset
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
                    # Normal RL inference — apply policy ramp if configured
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
                        print(f"[DIAG] #{inference_count:5d}  "
                              f"pitch={pitch_deg:+.1f}° roll={roll_deg:+.1f}°  "
                              f"q_err_max={q_err_abs[max_err_idx]:.3f}[{joint_names[max_err_idx]}]  "
                              f"tau_max={tau_max:.1f}  "
                              f"act_max={np.max(np.abs(action)):.3f}  "
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
        # Safe shutdown: transition to DAMP
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
