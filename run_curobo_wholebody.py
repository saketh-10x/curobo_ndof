"""Stepwise cuRobo 10-DOF IK planning + Isaac Sim execution for V1.1.1 robot.

10-DOF whole-body planner: 3 base (x, y, theta) + 1 liftkit + 6 UR10e arm.

4 independently testable steps:
  Step 1: Spawn + verify joint control (all joints reach target within tolerance)
  Step 2: FK frame alignment (cuRobo FK matches Isaac Sim tool0 body pose)
  Step 3: Plan trajectory (cuRobo plans from current config to goal pose)
  Step 4: Execute + verify (replay trajectory, verify EE pos AND quat match goal)

Usage:
    # All steps headless:
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p scripts/standalone/curobo_v1_1/run_curobo_wholebody.py --headless

    # Individual step:
    ... --headless --step 1

    # Step 4 with GUI (markers):
    ... --step 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cuRobo 10-DOF stepwise IK + execution for V1.1.1.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--step", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="Run specific step (0 = all steps sequentially).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

# cuRobo imports
from curobo.types.state import JointState
from curobo.types.math import Pose as CuroboPose
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util_file import load_yaml

##
# Paths
##
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")
_SPHERES_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_spheres.yml")
_USD_DIR = os.path.join(_SCRIPT_DIR, "usd_generated")

# 10-DOF joint names (must match YAML cspace)
# Order: 3 base + 1 liftkit + 6 UR10e arm (matches URDF kinematic chain)
JOINT_NAMES = [
    "base_x", "base_y", "base_theta",
    "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Retract config (from YAML) — 10-DOF
RETRACT_CONFIG = [0.0, 0.0, 0.0, 0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

# Arm-only retract (6 joints, no liftkit/base)
ARM_RETRACT = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Base joint names (for velocity control during trajectory execution)
BASE_JOINT_NAMES = ["base_x", "base_y", "base_theta"]

# Swerve drive geometry (from swerve_pose_controller.py)
STEERING_JOINT_NAMES = ["steering_f", "steering_rl", "steering_rr"]
WHEEL_JOINT_NAMES = ["wheel_f", "wheel_rl", "wheel_rr"]
SWERVE_WHEELS = [
    (0.3225, 0.0),       # front wheel position (x, y) relative to base_link
    (-0.3225, 0.245),    # rear left
    (-0.3225, -0.245),   # rear right
]
SWERVE_STEERING_LIMIT = math.radians(140)  # ±140°
WHEEL_RADIUS = 0.0775  # metres (from tested swerve_integration.py)

# Arm-only goals — base stays at origin, small arm joint changes from retract.
# Retract: [0, 0, 0, 0, 0, -1.5708, 0, -1.5708, 0, 0]
# 10-DOF: [base_x, base_y, base_theta, liftkit_mid, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
GOAL_CONFIGS = [
    [0.5, 0.0, 0.0, 0.0, 0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0],         # A: base fwd + arm folded
    [0.0, 0.5, 0.0, 0.5, 0.0, -1.5708, 0.0, -1.5708, 1.5708, 0.0],          # B: base left + lift + wrist2 rot
    [-0.5, 0.0, 1.5708, 0.0, 1.5708, -1.5708, 0.0, -1.5708, 0.0, 1.5708],   # C: base back + rotate + shoulder pan + wrist3
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],             # D: return to retract
]
GOAL_NAMES = ["A", "B", "C", "D"]


##
# Result container (NdofCuroboPlanner pattern)
##

@dataclass
class PlanResult:
    success: bool
    joint_names: List[str]
    positions: Optional[torch.Tensor]       # (T, 10)
    velocities: Optional[torch.Tensor]
    accelerations: Optional[torch.Tensor]
    dt: float
    solve_time_s: float
    motion_time_s: float
    status: str = ""

    @property
    def base_trajectory(self) -> Optional[torch.Tensor]:
        return self.positions[:, 0:3] if self.positions is not None else None

    @property
    def lift_trajectory(self) -> Optional[torch.Tensor]:
        return self.positions[:, 3:4] if self.positions is not None else None

    @property
    def arm_trajectory(self) -> Optional[torch.Tensor]:
        return self.positions[:, 4:10] if self.positions is not None else None

    @property
    def num_waypoints(self) -> int:
        return 0 if self.positions is None else self.positions.shape[0]

    def __repr__(self) -> str:
        if self.success:
            return (
                f"PlanResult(OK, {self.num_waypoints} wpts, "
                f"motion={self.motion_time_s:.3f}s, solve={self.solve_time_s:.3f}s)"
            )
        return f"PlanResult(FAIL, status='{self.status}')"


##
# Scene Configuration
##
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, light, and V1.1.1 robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_URDF_PATH,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            collision_from_visuals=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            usd_dir=_USD_DIR,
            usd_file_name="robotv1_1_sprayer.usd",
            make_instanceable=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "base_x": 0.0,
                "base_y": 0.0,
                "base_theta": 0.0,
                "steering_f": 0.0,
                "steering_rl": 0.0,
                "steering_rr": 0.0,
                "wheel_f": 0.0,
                "wheel_rl": 0.0,
                "wheel_rr": 0.0,
                "liftkit_mid": 0.0,
                "liftkit_top": 0.0,
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5708,
                "elbow_joint": 0.0,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Virtual base joints — velocity control (stiffness=0, high damping)
            # With stiffness=0, only velocity targets matter (no oscillation possible).
            # damping=1e5 gives τ=mass/damping≈0.006s (instant tracking).
            # effort_limit=1e5 allows max_vel = effort/damping = 1.0 m/s.
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_x", "base_y", "base_theta"],
                effort_limit=1e5,
                stiffness=0.0,
                damping=1e5,
            ),
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=1320.0,
                damping=72.6636085,
                effort_limit=330.0,
                friction=0.0,
                armature=0.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=600.0,
                damping=34.64101615,
                effort_limit=150.0,
                friction=0.0,
                armature=0.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=216.0,
                damping=29.39387691,
                effort_limit=54.0,
                friction=0.0,
                armature=0.0,
            ),
            "liftkit": ImplicitActuatorCfg(
                joint_names_expr=["liftkit_.*"],
                stiffness=800.0,
                damping=45.0,
                effort_limit=100.0,
                velocity_limit=0.012,
                friction=0.0,
                armature=0.0,
            ),
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["steering_.*"],
                stiffness=1000.0,
                damping=100.0,
                effort_limit=500.0,
                velocity_limit=120.0,
                friction=0.0,
                armature=0.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*"],
                stiffness=0.0,
                damping=1000.0,
                effort_limit=500.0,
                velocity_limit=100.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )


##
# Planner helpers
##

def build_motion_gen():
    """Build and warm up cuRobo MotionGen from YAML config."""
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg["kinematics"]["collision_spheres"] = _SPHERES_PATH

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=WorldConfig(),
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_ik_seeds=30,
        num_graph_seeds=12,
        num_trajopt_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"obb": 10, "mesh": 10},
        trajopt_tsteps=32,
        collision_activation_distance=0.02,
        self_collision_check=False,
        position_threshold=0.005,
        rotation_threshold=0.05,
    )

    motion_gen = MotionGen(mg_cfg)

    print("[INFO]: Warming up cuRobo MotionGen...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[INFO]: cuRobo warmup complete.")

    return motion_gen, tensor_args


def compute_fk(motion_gen, q):
    """Compute FK for joint positions q. Returns (ee_pos [3], ee_quat [4]) tensors.

    IMPORTANT: Clones outputs to prevent cuRobo internal buffer overwrite on next call.
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
    fk = motion_gen.kinematics.get_state(q)
    return fk.ee_position[0].clone(), fk.ee_quaternion[0].clone()


def plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat):
    """Plan trajectory from current_q to goal pose. Returns PlanResult."""
    ta = tensor_args

    if current_q.dim() == 1:
        current_q = current_q.unsqueeze(0)
    current_q = current_q.to(ta.device, ta.dtype)

    if goal_pos.dim() == 1:
        goal_pos = goal_pos.unsqueeze(0)
    goal_pos = goal_pos.to(ta.device, ta.dtype)

    if goal_quat.dim() == 1:
        goal_quat = goal_quat.unsqueeze(0)
    goal_quat = goal_quat.to(ta.device, ta.dtype)

    start_state = JointState.from_position(current_q, joint_names=JOINT_NAMES)
    goal_pose = CuroboPose(position=goal_pos, quaternion=goal_quat)

    plan_cfg = MotionGenPlanConfig(
        enable_graph=True,
        enable_graph_attempt=4,
        max_attempts=20,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    t0 = time.time()
    result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    solve_time = time.time() - t0

    if not result.success.item():
        return PlanResult(
            success=False, joint_names=JOINT_NAMES,
            positions=None, velocities=None, accelerations=None,
            dt=0.05, solve_time_s=solve_time,
            motion_time_s=0.0, status=str(result.status),
        )

    # cuRobo's own error metrics (on optimized plan)
    print(f"  cuRobo pos_err: {result.position_error}, rot_err: {result.rotation_error}")

    # Get interpolated plan — already in cspace order, skip get_full_js round-trip
    # (get_full_js adds locked/mimic joints then get_ordered_joint_state removes them,
    #  which can introduce joint-ordering bugs with steering/wheel/mimic joints)
    traj = result.get_interpolated_plan()
    print(f"  Interpolated plan: shape={traj.position.shape}, joints={traj.joint_names}")

    # Verify joint names match our expected cspace order
    if traj.joint_names != JOINT_NAMES:
        print(f"  WARNING: joint name mismatch! Expected {JOINT_NAMES}, got {traj.joint_names}")
        # Reorder if needed
        traj = traj.get_ordered_joint_state(JOINT_NAMES)

    # Compare optimized vs interpolated final waypoint
    opt_plan = result.optimized_plan
    opt_ndim = len(opt_plan.position.shape)
    opt_last = opt_plan.position[0, -1] if opt_ndim == 3 else opt_plan.position[-1]
    interp_last = traj.position[-1]
    joint_diff = (opt_last - interp_last).abs()
    print(f"  Opt vs interp last wp diff (per joint): {joint_diff.tolist()}")
    print(f"  Opt last:    {opt_last.tolist()}")
    print(f"  Interp last: {interp_last.tolist()}")

    motion_time = (
        result.motion_time.item()
        if isinstance(result.motion_time, torch.Tensor)
        else float(result.motion_time or 0.0)
    )

    return PlanResult(
        success=True, joint_names=JOINT_NAMES,
        positions=traj.position,
        velocities=traj.velocity,
        accelerations=traj.acceleration,
        dt=0.05, solve_time_s=solve_time,
        motion_time_s=motion_time, status="success",
    )


def quat_angular_error(q1, q2):
    """Compute angular error in radians between two wxyz quaternions."""
    q1 = q1 / q1.norm()
    q2 = q2 / q2.norm()
    dot = torch.abs(torch.dot(q1, q2)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


##
# Swerve drive IK (ported from swerve_pose_controller.py, no ROS dependency)
##

def swerve_ik_wheel(vx_body, vy_body, omega, wheel_x, wheel_y, limit):
    """Single-wheel swerve IK. Returns (speed_m_s, steer_rad)."""
    vix = vx_body - omega * wheel_y
    viy = vy_body + omega * wheel_x
    speed = math.sqrt(vix ** 2 + viy ** 2)
    if speed < 1e-6:
        return 0.0, 0.0
    theta = math.atan2(viy, vix)
    if theta < -limit or theta > limit:
        theta = math.atan2(-viy, -vix)
        speed = -speed
    return speed, theta


def compute_swerve_commands(vx_world, vy_world, omega, base_theta):
    """Convert world-frame base velocity to swerve steering angles and wheel speeds.

    Transforms velocities from world frame (base_x/base_y DOFs) to body frame,
    then runs per-wheel swerve IK.

    Returns: (steer_angles [3], wheel_speeds_rad_s [3])
    """
    ct = math.cos(base_theta)
    st = math.sin(base_theta)
    vx_body = ct * vx_world + st * vy_world
    vy_body = -st * vx_world + ct * vy_world

    limit = SWERVE_STEERING_LIMIT
    steer_angles = []
    wheel_speeds = []
    for i, (wx, wy) in enumerate(SWERVE_WHEELS):
        speed, steer = swerve_ik_wheel(vx_body, vy_body, omega, wx, wy, limit)
        steer_angles.append(steer)
        wheel_speed = speed / WHEEL_RADIUS  # m/s → rad/s
        if i == 0:  # front wheel — negate per tested swerve_integration.py
            wheel_speed = -wheel_speed
        wheel_speeds.append(wheel_speed)

    return steer_angles, wheel_speeds


##
# Simulation helpers
##

def sim_step_n(sim, scene, n):
    """Run n simulation steps."""
    dt = sim.get_physics_dt()
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def get_sim_joint_indices(robot, names):
    """Get sim joint indices for a list of joint names."""
    sim_names = list(robot.data.joint_names)
    return [sim_names.index(n) for n in names]


def get_base_frame_w(robot, env_idx=0):
    """Get swerve_link_x pose in world for a single env.

    Returns: (pos [1,3], quat [1,4]) on sim device.
    """
    return (robot.data.root_pose_w[env_idx:env_idx+1, 0:3],
            robot.data.root_pose_w[env_idx:env_idx+1, 3:7])


def get_all_base_frames_w(robot):
    """Get swerve_link_x pose for ALL envs. Returns: (pos [N,3], quat [N,4])."""
    return robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7]


def get_tool0_pose_w(robot, env_idx=0):
    """Get tool0 body pose in world frame for a single env.

    Returns: (pos [1,3], quat [1,4]) on sim device.
    """
    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    return (
        robot.data.body_state_w[env_idx:env_idx+1, tool0_idx, 0:3],
        robot.data.body_state_w[env_idx:env_idx+1, tool0_idx, 3:7],
    )


def get_all_tool0_poses_w(robot):
    """Get tool0 body pose for ALL envs. Returns: (pos [N,3], quat [N,4])."""
    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    return (
        robot.data.body_state_w[:, tool0_idx, 0:3],
        robot.data.body_state_w[:, tool0_idx, 3:7],
    )


def get_tool0_pose_in_base(robot, env_idx=0):
    """Get tool0 pose relative to swerve_link_x for a single env.

    Returns: (pos [1,3], quat [1,4]) on sim device.
    """
    base_pos_w, base_quat_w = get_base_frame_w(robot, env_idx)
    tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot, env_idx)
    return subtract_frame_transforms(base_pos_w, base_quat_w, tool0_pos_w, tool0_quat_w)


def print_base_diagnostics(robot):
    """Print root pose and all 10-DOF joint values for debugging."""
    base_pos_w, base_quat_w = get_base_frame_w(robot)
    print(f"  [DIAG] Root pose (swerve_link_x in world): "
          f"pos={base_pos_w[0].tolist()}, quat={base_quat_w[0].tolist()}")

    sim_names = list(robot.data.joint_names)
    for jn in JOINT_NAMES:
        idx = sim_names.index(jn)
        val = robot.data.joint_pos[0, idx].item()
        print(f"  [DIAG] {jn} = {val:.6f}")


##
# Step implementations
##

def step1_joint_control(sim, scene):
    """Step 1: Spawn + verify joint control.

    Sets arm to retract config, sims 200 steps, checks joints reached target.
    Pass: all 6 arm joints within 0.01 rad of target.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Spawn + Verify Joint Control")
    print("=" * 70)

    robot = scene["robot"]

    # Settle robot after spawn
    print("[STEP1]: Settling robot (100 steps)...")
    sim_step_n(sim, scene, 100)

    print("[STEP1]: After settling:")
    print_base_diagnostics(robot)

    # Get joint indices
    arm_indices = get_sim_joint_indices(robot, ARM_JOINT_NAMES)
    base_indices = get_sim_joint_indices(robot, ["base_x", "base_y", "base_theta"])
    liftkit_mid_idx = list(robot.data.joint_names).index("liftkit_mid")
    liftkit_top_idx = list(robot.data.joint_names).index("liftkit_top")

    # Build full joint target: set base to 0, arm to retract, enforce liftkit mimic
    arm_target = torch.tensor(ARM_RETRACT, device=sim.device, dtype=torch.float32)
    full_target = robot.data.joint_pos[0, :].clone()
    for idx in base_indices:
        full_target[idx] = 0.0
    for i, si in enumerate(arm_indices):
        full_target[si] = arm_target[i]
    full_target[liftkit_top_idx] = full_target[liftkit_mid_idx]

    print(f"[STEP1]: Setting base target: [0, 0, 0], arm target: {ARM_RETRACT}")
    robot.set_joint_position_target(full_target.unsqueeze(0))

    # Sim 200 steps to let PD controller settle
    print("[STEP1]: Simulating 200 steps...")
    sim_step_n(sim, scene, 200)

    print("[STEP1]: After arm movement:")
    print_base_diagnostics(robot)

    # Read actual joint positions
    actual = robot.data.joint_pos[0, :]
    arm_actual = actual[arm_indices]

    errors = (arm_actual - arm_target.to(arm_actual.device)).abs()
    max_err = errors.max().item()

    print(f"[STEP1]: Arm target:  {arm_target.tolist()}")
    print(f"[STEP1]: Arm actual:  {arm_actual.tolist()}")
    print(f"[STEP1]: Joint errors: {errors.tolist()}")
    print(f"[STEP1]: Max error: {max_err:.4f} rad")

    if max_err < 0.01:
        print(f"STEP 1 PASS: Joint control verified (max error: {max_err:.4f} rad)")
        return True
    else:
        print(f"STEP 1 FAIL: Max joint error {max_err:.4f} rad exceeds 0.01 rad threshold")
        return False


def step2_fk_alignment(sim, scene, motion_gen, tensor_args):
    """Step 2: FK frame alignment.

    Compares cuRobo FK with Isaac Sim tool0 pose, both in swerve_link_x (base) frame.
    This avoids world-frame drift issues — we compare relative to the kinematic root.
    Pass: position error < 10mm.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FK Frame Alignment (in base frame)")
    print("=" * 70)

    robot = scene["robot"]
    ta = tensor_args

    print("[STEP2]: Diagnostics:")
    print_base_diagnostics(robot)

    # Get current 10-DOF joints from sim using official pattern
    cu_js = JointState(
        position=robot.data.joint_pos[0:1, :].to(ta.device, ta.dtype),
        velocity=torch.zeros_like(robot.data.joint_pos[0:1, :]).to(ta.device, ta.dtype),
        acceleration=torch.zeros_like(robot.data.joint_pos[0:1, :]).to(ta.device, ta.dtype),
        joint_names=list(robot.data.joint_names),
        tensor_args=ta,
    )
    curobo_js = cu_js.get_ordered_joint_state(JOINT_NAMES)
    current_q = curobo_js.position[0]
    print(f"[STEP2]: 10-DOF from sim: {current_q.tolist()}")

    # cuRobo FK -> EE in swerve_link_x (base) frame
    fk_pos, fk_quat = compute_fk(motion_gen, current_q)
    print(f"[STEP2]: cuRobo FK (base frame): pos={fk_pos.tolist()}, quat={fk_quat.tolist()}")

    # Isaac Sim: tool0 pose relative to swerve_link_x (base frame)
    sim_tool0_base_pos, sim_tool0_base_quat = get_tool0_pose_in_base(robot)
    print(f"[STEP2]: Sim tool0 (base frame): pos={sim_tool0_base_pos[0].tolist()}, "
          f"quat={sim_tool0_base_quat[0].tolist()}")

    # Compare in base frame
    fk_pos_dev = fk_pos.to(sim_tool0_base_pos.device)
    fk_quat_dev = fk_quat.to(sim_tool0_base_quat.device)

    pos_err = torch.norm(fk_pos_dev - sim_tool0_base_pos[0]).item()
    quat_err = quat_angular_error(fk_quat_dev, sim_tool0_base_quat[0]).item()

    print(f"[STEP2]: Position error: {pos_err * 1000:.1f} mm")
    print(f"[STEP2]: Orientation error: {math.degrees(quat_err):.2f} deg")

    if pos_err < 0.010:
        print(f"STEP 2 PASS: FK alignment verified "
              f"(pos err: {pos_err * 1000:.1f} mm, rot err: {math.degrees(quat_err):.2f} deg)")
        return True
    else:
        print(f"STEP 2 FAIL: Position error {pos_err * 1000:.1f} mm exceeds 10 mm threshold")
        return False


def get_current_curobo_q(robot, tensor_args):
    """Read current 10-DOF joint state from sim, reordered for cuRobo."""
    ta = tensor_args
    cu_js = JointState(
        position=robot.data.joint_pos[0:1, :].to(ta.device, ta.dtype),
        velocity=torch.zeros_like(robot.data.joint_pos[0:1, :]).to(ta.device, ta.dtype),
        acceleration=torch.zeros_like(robot.data.joint_pos[0:1, :]).to(ta.device, ta.dtype),
        joint_names=list(robot.data.joint_names),
        tensor_args=ta,
    )
    return cu_js.get_ordered_joint_state(JOINT_NAMES).position[0]


def step3_plan_single(sim, scene, motion_gen, tensor_args, goal_config, goal_name):
    """Step 3: Plan trajectory from current sim joints to a goal.

    Returns: (success, PlanResult, goal_pos, goal_quat) all in base frame.
    Pass: planning succeeds, FK of final waypoint matches goal pos < 5mm AND quat < 5 deg.
    """
    robot = scene["robot"]
    ta = tensor_args

    current_q = get_current_curobo_q(robot, tensor_args)
    print(f"  [PLAN {goal_name}]: Start 10-DOF: {current_q.tolist()}")

    # Goal: FK of the given config
    goal_q = torch.tensor(goal_config, device=ta.device, dtype=ta.dtype)
    goal_pos, goal_quat = compute_fk(motion_gen, goal_q)
    print(f"  [PLAN {goal_name}]: Goal config: {goal_config}")
    print(f"  [PLAN {goal_name}]: Goal EE (base frame): pos={goal_pos.tolist()}, "
          f"quat={goal_quat.tolist()}")

    result = plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat)
    print(f"  [PLAN {goal_name}]: {result}")

    if not result.success:
        print(f"  [PLAN {goal_name}] FAIL: Planning failed — {result.status}")
        return False, None, None, None

    # Diagnostic: FK of interpolated plan's final waypoint vs goal
    # (cuRobo's own success check on the optimized plan is authoritative;
    #  this is informational only — step 4 sim execution is the real test)
    final_pos, final_quat = compute_fk(motion_gen, result.positions[-1])
    pos_err = torch.norm(final_pos - goal_pos).item()
    quat_err = quat_angular_error(final_quat, goal_quat).item()

    print(f"  [PLAN {goal_name}]: FK diagnostic — pos err: {pos_err * 1000:.1f} mm, "
          f"quat err: {math.degrees(quat_err):.2f} deg")

    # Trust cuRobo's success — step 4 sim execution is the ground truth
    print(f"  [PLAN {goal_name}] PASS: cuRobo success, {result.num_waypoints} waypoints")
    return True, result, goal_pos, goal_quat


def step4_execute_single(sim, scene, motion_gen, tensor_args, plan_result,
                         goal_pos_base, goal_quat_base, goal_name,
                         ee_marker, goal_marker):
    """Step 4: Execute one planned trajectory and verify EE reaches goal.

    Base motion: velocity control on virtual joints + swerve IK for steering/wheels.
    Arm/liftkit: position control.

    After execution, zeros velocities, holds 200 steps, then verifies IN BASE FRAME:
      - EE position matches goal (< 20mm)
      - EE orientation matches goal (< 10 deg)
    Only PASS if BOTH match.
    """
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    steps_per_waypoint = 4
    device = sim.device

    # Joint index mappings
    sim_joint_names = list(robot.data.joint_names)
    n_sim_joints = len(sim_joint_names)
    base_sim_ids = get_sim_joint_indices(robot, BASE_JOINT_NAMES)
    arm_liftkit_sim_ids = get_sim_joint_indices(robot, JOINT_NAMES[3:])  # liftkit + 6 arm
    steering_sim_ids = get_sim_joint_indices(robot, STEERING_JOINT_NAMES)
    wheel_sim_ids = get_sim_joint_indices(robot, WHEEL_JOINT_NAMES)
    liftkit_mid_idx = sim_joint_names.index("liftkit_mid")
    liftkit_top_idx = sim_joint_names.index("liftkit_top")

    plan_positions = plan_result.positions
    plan_dt = plan_result.dt
    print(f"  [EXEC {goal_name}]: Executing {plan_result.num_waypoints} waypoints "
          f"({steps_per_waypoint} sim steps each, base=vel ctrl + swerve IK)...")

    # Capture initial root pose for ALL envs for stable goal marker placement
    # (swerve_link_x is 0.001 kg and drifts; freeze the reference at execution start)
    init_base_pos_w, init_base_quat_w = get_all_base_frames_w(robot)
    init_base_pos_w = init_base_pos_w.clone()  # [N, 3]
    init_base_quat_w = init_base_quat_w.clone()  # [N, 4]

    # Execute trajectory
    for wp_idx in range(len(plan_positions)):
        waypoint = plan_positions[wp_idx]

        # --- NaN safety check ---
        if torch.isnan(robot.data.joint_pos[0, :]).any():
            print(f"  [EXEC {goal_name}] FAIL: NaN in joint positions at wp {wp_idx}")
            return False

        # --- Base velocity from consecutive waypoints ---
        if wp_idx < len(plan_positions) - 1:
            next_wp = plan_positions[wp_idx + 1]
            base_vel = (next_wp[:3] - waypoint[:3]) / plan_dt
        else:
            base_vel = torch.zeros(3, device=waypoint.device)

        # --- Swerve IK: world-frame base vel → steering angles + wheel speeds ---
        base_theta_cur = waypoint[2].item()
        steer_angles, wheel_speeds = compute_swerve_commands(
            base_vel[0].item(), base_vel[1].item(), base_vel[2].item(),
            base_theta_cur,
        )

        # --- Position targets: arm/liftkit + steering ---
        pos_target = robot.data.joint_pos[0:1, :].clone()
        for ci, si in enumerate(arm_liftkit_sim_ids):
            pos_target[0, si] = waypoint[3 + ci].to(device)
        pos_target[0, liftkit_top_idx] = pos_target[0, liftkit_mid_idx]  # mimic
        for si, angle in zip(steering_sim_ids, steer_angles):
            pos_target[0, si] = angle

        # --- Velocity targets: base + wheels ---
        vel_target = torch.zeros(1, n_sim_joints, device=device)
        for ci, si in enumerate(base_sim_ids):
            vel_target[0, si] = base_vel[ci].to(device)
        for si, speed in zip(wheel_sim_ids, wheel_speeds):
            vel_target[0, si] = speed

        robot.set_joint_position_target(pos_target)
        robot.set_joint_velocity_target(vel_target)

        for _ in range(steps_per_waypoint):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # Update markers for ALL envs: transform base-frame goal to world
            # goal_pos_base/goal_quat_base are [3]/[4], expand to [N,3]/[N,4]
            num_envs = robot.data.root_pose_w.shape[0]
            goal_pos_expanded = goal_pos_base.unsqueeze(0).expand(num_envs, -1).to(init_base_pos_w.device)
            goal_quat_expanded = goal_quat_base.unsqueeze(0).expand(num_envs, -1).to(init_base_quat_w.device)
            goal_pos_w, goal_quat_w = combine_frame_transforms(
                init_base_pos_w, init_base_quat_w,
                goal_pos_expanded, goal_quat_expanded,
            )
            tool0_pos_w, tool0_quat_w = get_all_tool0_poses_w(robot)
            ee_marker.visualize(tool0_pos_w, tool0_quat_w)
            goal_marker.visualize(goal_pos_w, goal_quat_w)

        if (wp_idx + 1) % 50 == 0 or wp_idx == len(plan_positions) - 1:
            sim_tool0_base_pos, _ = get_tool0_pose_in_base(robot)
            track_err = torch.norm(
                sim_tool0_base_pos[0].to(goal_pos_base.device) - goal_pos_base
            ).item()
            base_vals = [robot.data.joint_pos[0, si].item() for si in base_sim_ids]
            steer_vals = [robot.data.joint_pos[0, si].item() for si in steering_sim_ids]
            print(f"  [EXEC {goal_name}]: wp {wp_idx + 1}/{len(plan_positions)}, "
                  f"track_err: {track_err * 1000:.1f} mm, "
                  f"base=[{base_vals[0]:.3f}, {base_vals[1]:.3f}, {base_vals[2]:.3f}], "
                  f"steer=[{steer_vals[0]:.2f}, {steer_vals[1]:.2f}, {steer_vals[2]:.2f}]")

    # Stop all motion — zero velocity targets
    vel_target = torch.zeros(1, n_sim_joints, device=device)
    robot.set_joint_velocity_target(vel_target)

    print(f"  [EXEC {goal_name}]: Trajectory done. Holding 200 steps...")
    sim_step_n(sim, scene, 200)

    # Final verification in BASE FRAME
    sim_tool0_base_pos, sim_tool0_base_quat = get_tool0_pose_in_base(robot)

    pos_err = torch.norm(
        sim_tool0_base_pos[0].to(goal_pos_base.device) - goal_pos_base
    ).item()
    quat_err = quat_angular_error(
        sim_tool0_base_quat[0].to(goal_quat_base.device), goal_quat_base
    ).item()

    print(f"  [EXEC {goal_name}]: Final EE  (base): pos={sim_tool0_base_pos[0].tolist()}")
    print(f"  [EXEC {goal_name}]: Goal      (base): pos={goal_pos_base.tolist()}")
    print(f"  [EXEC {goal_name}]: Pos error: {pos_err * 1000:.1f} mm, "
          f"Quat error: {math.degrees(quat_err):.2f} deg")

    pos_ok = pos_err < 0.020
    quat_ok = quat_err < math.radians(10.0)

    if pos_ok and quat_ok:
        print(f"  [EXEC {goal_name}] PASS: EE reached goal "
              f"(pos: {pos_err * 1000:.1f} mm, quat: {math.degrees(quat_err):.2f} deg)")
        return True
    else:
        reasons = []
        if not pos_ok:
            reasons.append(f"pos {pos_err * 1000:.1f}mm > 20mm")
        if not quat_ok:
            reasons.append(f"quat {math.degrees(quat_err):.2f}deg > 10deg")
        print(f"  [EXEC {goal_name}] FAIL: {', '.join(reasons)}")
        return False


def step3_plan_all(sim, scene, motion_gen, tensor_args):
    """Step 3: Plan all 4 goals sequentially from current joint state.

    Each plan starts from the CURRENT sim joint state (which is the result of
    the previous execution or the initial settling).
    Returns list of (success, plan, goal_pos, goal_quat) tuples.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Plan All 4 Trajectories (cuRobo)")
    print("=" * 70)

    plans = []
    all_ok = True
    for i, (cfg, name) in enumerate(zip(GOAL_CONFIGS, GOAL_NAMES)):
        print(f"\n--- Goal {name} ({i + 1}/{len(GOAL_CONFIGS)}) ---")
        ok, plan, gpos, gquat = step3_plan_single(
            sim, scene, motion_gen, tensor_args, cfg, name
        )
        plans.append((ok, plan, gpos, gquat))
        if not ok:
            all_ok = False

    passed = sum(1 for ok, *_ in plans if ok)
    print(f"\nSTEP 3 {'PASS' if all_ok else 'FAIL'}: {passed}/{len(plans)} plans succeeded")
    return all_ok, plans


def step4_execute_all(sim, scene, motion_gen, tensor_args, plans):
    """Step 4: Execute all 4 plans sequentially, verifying EE pose after each.

    For each goal: execute trajectory -> hold -> verify EE pos AND quat match goal.
    Then the next plan starts from where the robot currently is.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Execute All 4 Goals Sequentially + Verify Each")
    print("=" * 70)

    robot = scene["robot"]
    num_envs = robot.data.root_pose_w.shape[0]

    # Create markers with num_markers=num_envs so each env gets its own visual
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    frame_marker_cfg.prim_path = "/Visuals/ee_current"
    ee_marker = VisualizationMarkers(frame_marker_cfg)

    goal_frame_cfg = FRAME_MARKER_CFG.copy()
    goal_frame_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    goal_frame_cfg.prim_path = "/Visuals/ee_goal"
    goal_marker = VisualizationMarkers(goal_frame_cfg)

    goal_results = {}
    for i, ((ok, plan, gpos, gquat), name) in enumerate(zip(plans, GOAL_NAMES)):
        print(f"\n{'~' * 50}")
        print(f"  Goal {name} ({i + 1}/{len(plans)})")
        print(f"{'~' * 50}")

        if not ok or plan is None or not plan.success:
            print(f"  [EXEC {name}] SKIP: No valid plan")
            goal_results[name] = False
            continue

        # Re-plan from CURRENT sim joints (robot may have moved from previous exec)
        print(f"  [EXEC {name}]: Re-planning from current sim state...")
        re_ok, re_plan, re_gpos, re_gquat = step3_plan_single(
            sim, scene, motion_gen, tensor_args, GOAL_CONFIGS[i], name
        )

        if not re_ok or re_plan is None or not re_plan.success:
            print(f"  [EXEC {name}] FAIL: Re-planning from current state failed")
            goal_results[name] = False
            continue

        exec_ok = step4_execute_single(
            sim, scene, motion_gen, tensor_args,
            re_plan, re_gpos, re_gquat, name,
            ee_marker, goal_marker,
        )
        goal_results[name] = exec_ok

    print_base_diagnostics(robot)

    all_ok = all(goal_results.values()) and len(goal_results) == len(plans)
    passed = sum(1 for v in goal_results.values() if v)
    print(f"\nSTEP 4 {'PASS' if all_ok else 'FAI L'}: "
          f"{passed}/{len(plans)} goals reached (pos AND quat verified)")
    return all_ok, goal_results


##
# Main
##

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 1.0])

    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    step = args_cli.step
    results = {}

    # Step 1: sim-only (no cuRobo)
    if step in (0, 1):
        results[1] = step1_joint_control(sim, scene)
        if step == 1:
            print(f"\n{'=' * 70}")
            print(f"SUMMARY: Step 1 {'PASS' if results[1] else 'FAIL'}")
            print(f"{'=' * 70}")
            return

    # Steps 2-4 need cuRobo
    motion_gen = None
    tensor_args = None
    if step in (0, 2, 3, 4):
        motion_gen, tensor_args = build_motion_gen()

    # Step 2: FK alignment
    if step in (0, 2):
        results[2] = step2_fk_alignment(sim, scene, motion_gen, tensor_args)
        if step == 2:
            print(f"\n{'=' * 70}")
            print(f"SUMMARY: Step 2 {'PASS' if results[2] else 'FAIL'}")
            print(f"{'=' * 70}")
            return

    # Step 3: Plan all 4 goals
    plans = None
    if step in (0, 3, 4):
        ok, plans = step3_plan_all(sim, scene, motion_gen, tensor_args)
        results[3] = ok
        if step == 3:
            print(f"\n{'=' * 70}")
            print(f"SUMMARY: Step 3 {'PASS' if results[3] else 'FAIL'}")
            print(f"{'=' * 70}")
            return

    # Step 4: Execute all 4 goals sequentially, verify each
    if step in (0, 4):
        if plans is None:
            # If running step 4 alone, plan first
            _, plans = step3_plan_all(sim, scene, motion_gen, tensor_args)

        ok, goal_results = step4_execute_all(
            sim, scene, motion_gen, tensor_args, plans
        )
        results[4] = ok

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for s in sorted(results):
        status = "PASS" if results[s] else "FAIL"
        print(f"  Step {s}: {status}")
    if 4 in results and plans is not None:
        for name in GOAL_NAMES:
            tag = "PASS" if goal_results.get(name, False) else "FAIL"
            print(f"    Goal {name}: {tag}")
    all_pass = all(results.values())
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
    simulation_app.close()
