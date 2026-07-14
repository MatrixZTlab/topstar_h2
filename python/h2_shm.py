import ctypes
import time

# Define standard integer types
c_uint8 = ctypes.c_uint8
c_uint16 = ctypes.c_uint16
c_uint32 = ctypes.c_uint32
c_uint64 = ctypes.c_uint64
c_float = ctypes.c_float
c_int16 = ctypes.c_int16
c_int32 = ctypes.c_int32

# Constants matching C++ definitions (constants.h + h2_robot_config.h)
H2_NUM_MOTORS   = 29
H2_NUM_MASTERS  = 4
H2_NUM_HANDS    = 2
HAND_DOF        = 6   # matches C++ HAND_DOF
HAND_SENSOR_CNT = 12  # matches C++ HAND_SENSOR_COUNT

# ── PS2 Key Bitmask Constants (matching ps2_joystick.h) ──────────────────────
PS2_KEY_R1     = (1 << 0)
PS2_KEY_L1     = (1 << 1)
PS2_KEY_START  = (1 << 2)
PS2_KEY_SELECT = (1 << 3)
PS2_KEY_R2     = (1 << 4)
PS2_KEY_L2     = (1 << 5)
PS2_KEY_A      = (1 << 8)   # Cross
PS2_KEY_B      = (1 << 9)   # Circle
PS2_KEY_X      = (1 << 10)  # Square
PS2_KEY_Y      = (1 << 11)  # Triangle
PS2_KEY_UP     = (1 << 12)
PS2_KEY_RIGHT  = (1 << 13)
PS2_KEY_DOWN   = (1 << 14)
PS2_KEY_LEFT   = (1 << 15)

# ── FSM State Constants (matching h2_fsm.h) ──────────────────────────────────
FSM_ZERO_TORQUE  = 0
FSM_DAMP         = 1
FSM_SQUAT        = 2
FSM_SIT          = 3
FSM_STAND_UP     = 4
FSM_RL_INFERENCE = 10
FSM_LOCOMOTION   = 500

# ── DDS Service Constants (matching h2_service_types.h) ──────────────────────
MOTION_SWITCHER_API_ID_SELECT_MODE  = 1002
MOTION_SWITCHER_API_ID_RELEASE_MODE = 1003
LOCO_API_ID_SET_VELOCITY = 7105

# ── Shared Memory Structures (matching ec_shared_mem.h with #pragma pack(push,1)) ──

class MotorData(ctypes.Structure):
    """Matches C++ MotorData (pack 1, 58 bytes).
    Python field names preserve the API used by g1_rl_runner.py:
      q      → position    (offset 4)
      dq     → velocity    (offset 8)
      tau_est→ torque      (offset 12)
      q_des  → target_pos  (offset 26)
      dq_des → target_vel  (offset 30)
      tau_ff → target_tff  (offset 34)
      kp     → KP          (offset 38)
      kd     → KD          (offset 42)
      enabled→ enabled     (offset 46)
    """
    _pack_ = 1
    _fields_ = [
        ("mode",         c_uint8),      # C++ motor_type, offset 0
        ("master_idx",   c_uint8),      # offset 1
        ("slave_idx",    c_uint8),      # offset 2
        ("joint_id",     c_uint8),      # offset 3
        ("q",            c_float),      # C++ position,   offset 4
        ("dq",           c_float),      # C++ velocity,   offset 8
        ("tau_est",      c_float),      # C++ torque,     offset 12
        ("state",        c_uint16),     # offset 16
        ("error",        c_uint8),      # offset 18
        ("temperature",  c_int16),      # offset 19
        ("voltage",      c_float),      # offset 21
        ("body_part",    c_uint8),      # offset 25
        ("q_des",        c_float),      # C++ target_pos, offset 26
        ("dq_des",       c_float),      # C++ target_vel, offset 30
        ("tau_ff",       c_float),      # C++ target_tff, offset 34
        ("kp",           c_float),      # C++ KP,         offset 38
        ("kd",           c_float),      # C++ KD,         offset 42
        ("enabled",      c_uint8),      # offset 46
        ("new_command",  c_uint8),      # offset 47
        ("_pad48",       c_uint8 * 2),  # offset 48: zero_encoder_cycles + padding
        ("timestamp_ns", c_uint64),     # offset 50
    ]
    # Total: 58 bytes

class IMUData(ctypes.Structure):
    """Matches C++ IMUData (pack 1, 72 bytes).
    quaternion and gyroscope are at the same offsets as before.
    """
    _pack_ = 1
    _fields_ = [
        ("quaternion",    c_float * 4),   # w,x,y,z  offset 0
        ("gyroscope",     c_float * 3),   # x,y,z    offset 16
        ("accelerometer", c_float * 3),   # x,y,z    offset 28
        ("rpy",           c_float * 3),   # r,p,y    offset 40
        ("temperature",   c_int16),       # C++ int16_t, offset 52
        ("_pad54",        c_uint8 * 2),   # offset 54
        ("timestamp_ns",  c_uint64),      # offset 56
        ("valid",         c_uint8),       # offset 64
        ("_pad65",        c_uint8 * 7),   # offset 65
    ]
    # Total: 72 bytes

class JoystickData(ctypes.Structure):
    """Matches C++ JoystickData (pack 1, 36 bytes).
    Without _pack_=1 ctypes inserts 4 implicit bytes before timestamp_ns,
    pushing valid from offset 28 to 32 — always reading zero from C++ padding.
    """
    _pack_ = 1
    _fields_ = [
        ("lx",           c_float),      # offset 0
        ("ly",           c_float),      # offset 4
        ("rx",           c_float),      # offset 8
        ("ry",           c_float),      # offset 12
        ("keys",         c_uint16),     # offset 16
        ("_pad18",       c_uint8 * 2),  # offset 18
        ("timestamp_ns", c_uint64),     # offset 20  (pack 1 — no implicit gap)
        ("valid",        c_uint8),      # offset 28
        ("_pad29",       c_uint8 * 7),  # offset 29
    ]
    # Total: 36 bytes

class HandData(ctypes.Structure):
    """Opaque 298-byte blob matching C++ HandData (pack 1).
    The RL runner never touches hand data; the blob just preserves offsets.
    """
    _pack_ = 1
    _fields_ = [("_blob", c_uint8 * 298)]

class BmsData(ctypes.Structure):
    """Matches C++ BmsData (pack 1, 74 bytes).
    Battery Management System data from Greenway BMS via NiRen CAN-USB converter.
    Updated by topstar_bridge.run_bms(); published to rt/bms/state.
    """
    _pack_ = 1
    _fields_ = [
        ("soc",            c_uint8),       # State of charge 0-100%, offset 0
        ("status",         c_uint8 * 3),   # Protection flags (state1/2/3), offset 1
        ("current",        c_int32),       # Current in mA (pos=charging), offset 4
        ("pack_voltage",   c_uint32),      # Pack voltage in mV, offset 8
        ("remain_cap",     c_uint32),      # Remaining capacity in mAh, offset 12
        ("full_cap",       c_uint32),      # Full charge capacity in mAh, offset 16
        ("cycle",          c_uint16),      # Charge cycle count, offset 20
        ("temperature",    c_int16 * 2),   # NTC temps in °C (ntc1, ntc2), offset 22
        ("cell_vol",       c_uint16 * 16), # Cell voltages in mV (16 cells), offset 26
        ("valid",          c_uint8),       # Data has been received, offset 58
        ("_pad59",         c_uint8 * 7),   # Alignment, offset 59
        ("timestamp_ns",   c_uint64),      # Last update timestamp, offset 66
    ]
    # Total: 74 bytes

class LocoState(ctypes.Structure):
    """Matches C++ LocoState (pack 1, 116 bytes).
    Added missing fields: balance_mode, stand_height, vel_duration,
    task_id, speed_mode, arm_task_active, arm_sdk_active.
    fsm_id and velocity are at the same offsets.
    """
    _pack_ = 1
    _fields_ = [
        ("fsm_id",       c_int32),      # offset 0
        ("fsm_mode",     c_int32),      # offset 4
        ("balance_mode", c_int32),      # offset 8  (was absent in old Python)
        ("swing_height", c_float),      # offset 12
        ("stand_height", c_float),      # offset 16 (was absent in old Python)
        ("velocity",     c_float * 3),  # offset 20
        ("vel_duration", c_float),      # offset 32 (was absent)
        ("task_id",      c_int32),      # offset 36 (was absent)
        ("speed_mode",      c_int32),      # offset 40
        ("arm_task_active", c_int32),      # offset 44
        ("arm_sdk_active",     c_int32),      # offset 48
        ("timestamp",          c_uint64),     # offset 52
        ("_arm_gesture_pad",   c_uint8 * 24), # offset 60  (arm gesture floats, opaque)
        ("motor_fault_active", c_uint8),      # offset 84
        ("faulted_motor_id",   c_uint8),      # offset 85
        ("_pad",               c_uint8 * 30), # offset 86
    ]
    # Total: 116 bytes

class SharedMemoryData(ctypes.Structure):
    """Matches C++ SharedMemoryData (pack 1).
    Field order: motors → imu → joystick → hands → loco → bms → state_seq
    """
    _pack_ = 1
    _fields_ = [
        # ── Header (52 bytes) ──────────────────────────────────────────
        ("num_motors",         c_int32),          # offset 0
        ("master_count",       c_int32),          # offset 4
        ("motors_per_master",  c_int32 * 4),      # offset 8  (16 bytes)
        ("rt_thread_running",  c_uint8),           # offset 24
        ("_pad1",              c_uint8 * 3),       # offset 25
        ("cycle_counter",      c_uint64),          # offset 28
        ("last_cycle_time_ns", c_uint64),          # offset 36
        ("max_cycle_time_ns",  c_uint64),          # offset 44
        # ── Data ──────────────────────────────────────────────────────
        ("motors",   MotorData * H2_NUM_MOTORS),   # offset 52  (29×52 = 1508 B)
        ("imu",      IMUData),                     # offset 1560 (72 B)
        ("joystick", JoystickData),                # offset 1632 (36 B)
        ("hands",    HandData * H2_NUM_HANDS),     # offset 1668 (2×298 = 596 B)
        ("loco",     LocoState),                   # offset 2264 (116 B)
        ("bms",      BmsData),                     # offset 2380 (72 B)
        # Fix 2 (src_v2): state_seq seqlock added before the tail padding.
        # Odd value = RT thread is writing motor states; even = data stable.
        ("state_seq", c_uint32),                   # offset 2452 (4 B)
        ("_pad456",  c_uint8 * 252),               # offset 2456 — matches C padding[252]
    ]
    # Total: 2710 bytes (matches src_v2/ec_shared_mem.h)

# ── Compatibility Classes for g1_rl_runner.py ────────────────────────────────

class MotorState:
    def __init__(self):
        self.q = 0.0
        self.dq = 0.0
        self.tau_est = 0.0
        self.mode = 0

class IMUState:
    def __init__(self):
        self.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]
        self.accelerometer = [0.0, 0.0, 0.0]
        self.rpy = [0.0, 0.0, 0.0]

class JoystickState:
    """Python-friendly PS2 joystick state."""
    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.keys = 0
        self.valid = False

    def button_pressed(self, key_mask):
        """Check if a button (PS2_KEY_*) is currently held."""
        return bool(self.keys & key_mask)

class LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H2_NUM_MOTORS)]
        self.imu_state = IMUState()
        self.tick = 0
        self.motor_fault_active = False  # True if FSM_DAMP was triggered by a fault
        self.faulted_motor_id = 0xFF     # Joint ID of first faulted motor

class MotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0

class LowCmd:
    def __init__(self):
        self.motor_cmd = [MotorCmd() for _ in range(H2_NUM_MOTORS)]


# ── Shared Memory Interface ───────────────────────────────────────────────────

SHM_NAME = "/ec_motor_shm"
SEM_NAME = "/ec_motor_sem"

class H2SharedMemory:
    def __init__(self):
        import posix_ipc
        import mmap

        self.shm_fd = posix_ipc.SharedMemory(SHM_NAME)
        self.mem = mmap.mmap(self.shm_fd.fd, ctypes.sizeof(SharedMemoryData))
        self.shm = SharedMemoryData.from_buffer(self.mem)

        try:
            self.sem = posix_ipc.Semaphore(SEM_NAME)
        except posix_ipc.ExistentialError:
            print("Warning: Semaphore not found. Creating...")
            self.sem = posix_ipc.Semaphore(SEM_NAME, flags=posix_ipc.O_CREAT,
                                           initial_value=1)

    def close(self):
        self.mem.close()
        self.shm_fd.close()

    def get_low_state(self) -> LowState:
        """Read motor + IMU state from shared memory using seqlock (Fix 2/5).

        Retries if the RT thread is mid-write (state_seq is odd or changed).
        This never blocks the RT EtherCAT thread — no semaphore is held.
        """
        state = LowState()

        while True:
            seq = self.shm.state_seq
            if seq & 1:
                # RT thread is writing — spin briefly until stable
                continue

            state.imu_state.quaternion    = list(self.shm.imu.quaternion)
            state.imu_state.gyroscope     = list(self.shm.imu.gyroscope)
            state.imu_state.accelerometer = list(self.shm.imu.accelerometer)
            state.imu_state.rpy           = list(self.shm.imu.rpy)

            for i in range(H2_NUM_MOTORS):
                state.motor_state[i].q       = self.shm.motors[i].q
                state.motor_state[i].dq      = self.shm.motors[i].dq
                state.motor_state[i].tau_est = self.shm.motors[i].tau_est
                state.motor_state[i].mode    = self.shm.motors[i].mode

            state.motor_fault_active = bool(self.shm.loco.motor_fault_active)
            state.faulted_motor_id   = self.shm.loco.faulted_motor_id

            if self.shm.state_seq == seq:
                break  # snapshot was consistent

        return state

    def set_low_cmd(self, cmd: LowCmd):
        """Write motor commands to shared memory and update the watchdog timestamp."""
        self.shm.loco.timestamp = time.monotonic_ns()

        for i in range(H2_NUM_MOTORS):
            if cmd.motor_cmd[i].kp > 0 or cmd.motor_cmd[i].kd > 0:
                self.shm.motors[i].q_des   = cmd.motor_cmd[i].q
                self.shm.motors[i].dq_des  = cmd.motor_cmd[i].dq
                self.shm.motors[i].tau_ff  = cmd.motor_cmd[i].tau
                self.shm.motors[i].kp          = cmd.motor_cmd[i].kp
                self.shm.motors[i].kd          = cmd.motor_cmd[i].kd
                self.shm.motors[i].enabled     = 1
                self.shm.motors[i].new_command = 1  # required for LS (arm/waist) motors

    def get_joystick_state(self) -> JoystickState:
        """Read PS2 joystick state from shared memory."""
        js = JoystickState()
        js.lx    = self.shm.joystick.lx
        js.ly    = self.shm.joystick.ly
        js.rx    = self.shm.joystick.rx
        js.ry    = self.shm.joystick.ry
        js.keys  = self.shm.joystick.keys
        js.valid = bool(self.shm.joystick.valid)
        return js

    def get_fsm_id(self) -> int:
        """Get current FSM state ID from shared memory."""
        return self.shm.loco.fsm_id

    def set_fsm_id(self, fsm_id: int):
        """Write FSM state ID to shared memory (fallback when DDS unavailable)."""
        self.shm.loco.fsm_id = fsm_id

    def get_motor_fault(self) -> tuple:
        """Return (fault_active: bool, faulted_joint_id: int) from shared memory.

        faulted_joint_id is 0xFF when no fault is active.
        """
        return (bool(self.shm.loco.motor_fault_active),
                self.shm.loco.faulted_motor_id)

    def set_loco_velocity(self, vx: float, vy: float, vyaw: float):
        """Write velocity command to shared memory."""
        self.shm.loco.velocity[0] = vx
        self.shm.loco.velocity[1] = vy
        self.shm.loco.velocity[2] = vyaw


# ── PS4 Controller Direct Reader ──────────────────────────────────────────────
# Reads /dev/input/js1 using the Linux joystick API and maps PS4 buttons to
# the same PS2_KEY_* bitmasks as the C ps4_joystick.c driver, so it is a
# drop-in replacement for H2SharedMemory.get_joystick_state().

import struct as _struct
import fcntl as _fcntl
import os as _os

_JS_EVENT_FMT  = "IhBB"   # uint32 time, int16 value, uint8 type, uint8 number
_JS_EVENT_SIZE = _struct.calcsize(_JS_EVENT_FMT)
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS   = 0x02
_JS_EVENT_INIT   = 0x80

# PS4 button index → PS2_KEY_* bitmask (mirrors ps4_joystick.c ps4_map_button)
_PS4_BTN_TO_PS2 = {
    0: PS2_KEY_A,       # Cross
    1: PS2_KEY_B,       # Circle
    2: PS2_KEY_Y,       # Triangle
    3: PS2_KEY_X,       # Square
    4: PS2_KEY_L1,
    5: PS2_KEY_R1,
    6: PS2_KEY_L2,
    7: PS2_KEY_R2,
    8: PS2_KEY_SELECT,  # Share
    9: PS2_KEY_START,   # Options
}
_PS4_DPAD_THRESHOLD = 16000  # raw int16 value (matches C #define)

# Nintendo Switch Pro Controller button index → PS2_KEY_* bitmask
# (mirrors ps4_joystick.c sw_map_button; face buttons by printed label)
_SW_BTN_TO_PS2 = {
    0:  PS2_KEY_B,      # B (south)
    1:  PS2_KEY_A,      # A (east)
    2:  PS2_KEY_X,      # X (north)
    3:  PS2_KEY_Y,      # Y (west)
    5:  PS2_KEY_L1,     # L
    6:  PS2_KEY_R1,     # R
    7:  PS2_KEY_L2,     # ZL
    8:  PS2_KEY_R2,     # ZR
    9:  PS2_KEY_SELECT, # minus
    10: PS2_KEY_START,  # plus
}

_JSIOCGNAME = 0x80006a13 | (0x40 << 16)  # JSIOCGNAME(64)

class PS4JoystickReader:
    """Read a PS4 or Switch Pro controller directly from /dev/input/js*.

    Produces the same JoystickState layout (lx/ly/rx/ry + PS2-compatible
    keys bitmask) as H2SharedMemory.get_joystick_state(), so the two are
    interchangeable in h2_rl_runner.py.

    PS4 axis layout (verified via ps4_joystick.h):
      0=LX  1=LY  2=L2  3=RX  4=RY  5=R2  6=DPAD_X  7=DPAD_Y
    Switch Pro (hid-nintendo, auto-detected by device name):
      0=LX  1=LY  2=RX  3=RY  4=DPAD_X  5=DPAD_Y  (ZL/ZR digital only)
    """

    DEFAULT_DEVICE = "/dev/input/js1"

    def __init__(self, device: str = None):
        dev = device or self.DEFAULT_DEVICE
        self._fd = open(dev, "rb", buffering=0)
        fl = _fcntl.fcntl(self._fd, _fcntl.F_GETFL)
        _fcntl.fcntl(self._fd, _fcntl.F_SETFL, fl | _os.O_NONBLOCK)
        self._axis    = [0] * 8
        self._buttons = [0] * 16
        buf = bytearray(64)
        try:
            _fcntl.ioctl(self._fd, _JSIOCGNAME, buf)
        except OSError:
            pass
        name = bytes(buf).split(b"\0")[0].decode(errors="replace")
        self._is_switch = ("Nintendo" in name) or ("Pro Controller" in name)
        print(f"[PS4] Opened {dev} ({name})")
        if self._is_switch:
            print("[PS4] Using Nintendo Switch Pro Controller mapping")

    def read(self) -> JoystickState:
        """Drain pending events and return current JoystickState."""
        while True:
            try:
                raw = self._fd.read(_JS_EVENT_SIZE)
                if not raw or len(raw) < _JS_EVENT_SIZE:
                    break
                _, value, ev_type, number = _struct.unpack(_JS_EVENT_FMT, raw)
                ev_type &= ~_JS_EVENT_INIT
                if ev_type == _JS_EVENT_AXIS and number < 8:
                    self._axis[number] = value
                elif ev_type == _JS_EVENT_BUTTON and number < 16:
                    self._buttons[number] = value
            except BlockingIOError:
                break

        js = JoystickState()
        js.lx = self._axis[0] / 32767.0   # LX
        js.ly = self._axis[1] / 32767.0   # LY
        if self._is_switch:
            js.rx = self._axis[2] / 32767.0   # RX
            js.ry = self._axis[3] / 32767.0   # RY
            btn_table = _SW_BTN_TO_PS2
            dpad_x, dpad_y = self._axis[4], self._axis[5]
        else:
            js.rx = self._axis[3] / 32767.0   # RX
            js.ry = self._axis[4] / 32767.0   # RY
            btn_table = _PS4_BTN_TO_PS2
            dpad_x, dpad_y = self._axis[6], self._axis[7]

        keys = 0
        for btn_idx, ps2_key in btn_table.items():
            if btn_idx < len(self._buttons) and self._buttons[btn_idx]:
                keys |= ps2_key
        if dpad_x < -_PS4_DPAD_THRESHOLD:
            keys |= PS2_KEY_LEFT
        elif dpad_x > _PS4_DPAD_THRESHOLD:
            keys |= PS2_KEY_RIGHT
        if dpad_y < -_PS4_DPAD_THRESHOLD:
            keys |= PS2_KEY_UP
        elif dpad_y > _PS4_DPAD_THRESHOLD:
            keys |= PS2_KEY_DOWN

        js.keys  = keys
        js.valid = True
        return js

    def close(self):
        self._fd.close()
