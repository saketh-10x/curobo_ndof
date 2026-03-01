"""Stepwise cuRobo 7-DOF IK planning + Isaac Sim execution for V1.1.1 robot.

7-DOF planner: 1 liftkit + 6 UR10e arm (base locked at origin).

3 independently testable steps:
  Step 1: Spawn + verify joint control (arm reaches retract config)
  Step 2: FK frame alignment (cuRobo FK matches Isaac Sim tool0 in base frame)
  Step 3: Plan + execute 4 goals sequentially, verify EE pose

Usage:
    # All steps headless:
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p scripts/standalone/curobo_v1_1/run_curobo_7dof.py --headless

    # Individual step:
    ... --headless --step 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cuRobo 7-DOF stepwise IK + execution for V1.1.1.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--step", type=int, default=0, choices=[0, 1, 2, 3],
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
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
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

# Local robot config
from robot_cfg import ROBOT_V1_1_CFG

# Local paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")
_SPHERES_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_spheres.yml")

# 7-DOF joint names: liftkit + 6 UR10e arm (must match YAML cspace minus base joints)
JOINT_NAMES = [
    "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Arm-only joint names (6 UR10e joints, no liftkit)
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Retract config — 7-DOF [liftkit, pan, lift, elbow, w1, w2, w3]
RETRACT_CONFIG = [0.0, 0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]

# Arm-only retract (6 joints, no liftkit)
ARM_RETRACT = [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]

# Goal configs — 7-DOF [liftkit, pan, lift, elbow, w1, w2, w3]
GOAL_CONFIGS = [
    [0.0, 0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0],   # A: arm folded up
    [0.0, 1.0, -1.0, 0.5, -1.0, 0.0, 0.0],              # B: arm to the side
    [0.3, 0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0],     # C: liftkit raised
    [0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],        # D: retract (home)
]
GOAL_NAMES = ["A", "B", "C", "D"]


##
# Result container
##

@dataclass
class PlanResult:
    success: bool
    joint_names: List[str]
    positions: Optional[torch.Tensor]       # (T, 7)
    velocities: Optional[torch.Tensor]
    accelerations: Optional[torch.Tensor]
    dt: float
    solve_time_s: float
    motion_time_s: float
    status: str = ""

    @property
    def lift_trajectory(self) -> Optional[torch.Tensor]:
        return self.positions[:, 0:1] if self.positions is not None else None

    @property
    def arm_trajectory(self) -> Optional[torch.Tensor]:
        return self.positions[:, 1:7] if self.positions is not None else None

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
    """Scene with ground plane, light, and V1.1.1 robot (high-stiffness config)."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = ROBOT_V1_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# Planner helpers
##

def build_motion_gen():
    """Build and warm up cuRobo MotionGen from YAML config (no collision meshes)."""
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg["kinematics"]["collision_spheres"] = _SPHERES_PATH

    # Lock the 3 virtual base joints at 0 — reduces cspace from 10-DOF to 7-DOF
    lock_joints = robot_cfg["kinematics"].get("lock_joints", {})
    lock_joints["base_x"] = 0.0
    lock_joints["base_y"] = 0.0
    lock_joints["base_theta"] = 0.0
    robot_cfg["kinematics"]["lock_joints"] = lock_joints

    # Update cspace to 7-DOF (remove base joints)
    robot_cfg["kinematics"]["cspace"]["joint_names"] = list(JOINT_NAMES)
    robot_cfg["kinematics"]["cspace"]["retract_config"] = list(RETRACT_CONFIG)
    robot_cfg["kinematics"]["cspace"]["null_space_weight"] = [8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    robot_cfg["kinematics"]["cspace"]["cspace_distance_weight"] = [8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=WorldConfig(),
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_ik_seeds=30,
        num_graph_seeds=12,
        num_trajopt_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"mesh": 10, "obb": 10},
        trajopt_tsteps=32,
        collision_activation_distance=0.02,
        self_collision_check=False,
        position_threshold=0.005,
        rotation_threshold=0.05,
    )

    motion_gen = MotionGen(mg_cfg)
    print("[INFO]: Warming up cuRobo MotionGen...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[INFO]: Warmup complete.")

    return motion_gen, tensor_args


def compute_fk(motion_gen, q):
    """Compute FK for 7-DOF joint positions q. Returns (ee_pos [3], ee_quat [4]) tensors.

    Clones outputs to prevent cuRobo internal buffer overwrite on next call.
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
    fk = motion_gen.kinematics.get_state(q)
    return fk.ee_position[0].clone(), fk.ee_quaternion[0].clone()


def plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat):
    """Plan trajectory from current_q (7-DOF) to goal pose. Returns PlanResult."""
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

    traj = result.get_interpolated_plan()
    full_js = motion_gen.get_full_js(traj)
    ordered = full_js.get_ordered_joint_state(JOINT_NAMES)

    motion_time = (
        result.motion_time.item()
        if isinstance(result.motion_time, torch.Tensor)
        else float(result.motion_time or 0.0)
    )

    return PlanResult(
        success=True, joint_names=JOINT_NAMES,
        positions=ordered.position,
        velocities=ordered.velocity,
        accelerations=ordered.acceleration,
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


def get_tool0_pose_in_base(robot, env_idx=0):
    """Get tool0 pose relative to swerve_link_x (base) for a single env.

    Returns: (pos [1,3], quat [1,4]) on sim device.
    """
    base_pos_w = robot.data.root_pose_w[env_idx:env_idx+1, 0:3]
    base_quat_w = robot.data.root_pose_w[env_idx:env_idx+1, 3:7]

    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    tool0_pos_w = robot.data.body_state_w[env_idx:env_idx+1, tool0_idx, 0:3]
    tool0_quat_w = robot.data.body_state_w[env_idx:env_idx+1, tool0_idx, 3:7]

    return subtract_frame_transforms(base_pos_w, base_quat_w, tool0_pos_w, tool0_quat_w)


def get_all_tool0_poses_w(robot):
    """Get tool0 body pose in world frame for ALL envs. Returns: (pos [N,3], quat [N,4])."""
    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    return (
        robot.data.body_state_w[:, tool0_idx, 0:3],
        robot.data.body_state_w[:, tool0_idx, 3:7],
    )


def get_all_base_frames_w(robot):
    """Get swerve_link_x pose for ALL envs. Returns: (pos [N,3], quat [N,4])."""
    return robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7]


def get_current_curobo_q(robot, tensor_args):
    """Read current 7-DOF joint state from sim, reordered for cuRobo."""
    ta = tensor_args
    cu_js = JointState(
        position=robot.data.joint_pos[0:1, :].to(ta.device, ta.dtype),
        velocity=torch.zeros_like(robot.data.joint_pos[0:1, :]).to(ta.device, ta.dtype),
        acceleration=torch.zeros_like(robot.data.joint_pos[0:1, :]).to(ta.device, ta.dtype),
        joint_names=list(robot.data.joint_names),
        tensor_args=ta,
    )
    return cu_js.get_ordered_joint_state(JOINT_NAMES).position[0]


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

    # Get joint indices
    arm_indices = get_sim_joint_indices(robot, ARM_JOINT_NAMES)
    liftkit_mid_idx = list(robot.data.joint_names).index("liftkit_mid")
    liftkit_top_idx = list(robot.data.joint_names).index("liftkit_top")

    # Build full joint target: arm to retract, enforce liftkit mimic
    arm_target = torch.tensor(ARM_RETRACT, device=sim.device, dtype=torch.float32)
    full_target = robot.data.joint_pos[0, :].clone()
    for i, si in enumerate(arm_indices):
        full_target[si] = arm_target[i]
    full_target[liftkit_top_idx] = full_target[liftkit_mid_idx]

    print(f"[STEP1]: Setting arm target: {ARM_RETRACT}")
    robot.set_joint_position_target(full_target.unsqueeze(0))

    # Sim 200 steps to let PD controller settle
    print("[STEP1]: Simulating 200 steps...")
    sim_step_n(sim, scene, 200)

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

    Compares cuRobo FK with Isaac Sim tool0 pose, both in base frame.
    Pass: position error < 10mm.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FK Frame Alignment (in base frame)")
    print("=" * 70)

    robot = scene["robot"]

    # Get current 7-DOF joints from sim
    current_q = get_current_curobo_q(robot, tensor_args)
    print(f"[STEP2]: 7-DOF from sim: {current_q.tolist()}")

    # cuRobo FK -> EE in base frame
    fk_pos, fk_quat = compute_fk(motion_gen, current_q)
    print(f"[STEP2]: cuRobo FK (base frame): pos={fk_pos.tolist()}, quat={fk_quat.tolist()}")

    # Isaac Sim: tool0 pose relative to base
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


def step3_plan_and_execute(sim, scene, motion_gen, tensor_args):
    """Step 3: Plan + execute 4 goals sequentially, verify EE pose after each.

    For each goal:
      1. Compute goal EE pose via FK of goal config
      2. Plan from current sim joints to goal pose
      3. Execute trajectory (position control only, no base motion)
      4. Verify EE position < 20mm and orientation < 10 deg

    Pass: all 4 goals reached within tolerance.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Plan + Execute 4 Goals Sequentially")
    print("=" * 70)

    robot = scene["robot"]
    ta = tensor_args
    device = sim.device
    steps_per_waypoint = 4

    # Joint index mappings
    sim_joint_names = list(robot.data.joint_names)
    arm_liftkit_sim_ids = get_sim_joint_indices(robot, JOINT_NAMES)  # liftkit + 6 arm
    liftkit_mid_idx = sim_joint_names.index("liftkit_mid")
    liftkit_top_idx = sim_joint_names.index("liftkit_top")

    # Create visualization markers for EE and goal
    ee_marker_cfg = FRAME_MARKER_CFG.copy()
    ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker_cfg.prim_path = "/Visuals/ee_current"
    ee_marker = VisualizationMarkers(ee_marker_cfg)

    goal_marker_cfg = FRAME_MARKER_CFG.copy()
    goal_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    goal_marker_cfg.prim_path = "/Visuals/ee_goal"
    goal_marker = VisualizationMarkers(goal_marker_cfg)

    goal_results = {}

    for gi, (goal_config, goal_name) in enumerate(zip(GOAL_CONFIGS, GOAL_NAMES)):
        print(f"\n{'~' * 50}")
        print(f"  Goal {goal_name} ({gi + 1}/{len(GOAL_CONFIGS)})")
        print(f"{'~' * 50}")

        # Read current 7-DOF from sim
        current_q = get_current_curobo_q(robot, tensor_args)
        print(f"  [GOAL {goal_name}]: Start 7-DOF: {current_q.tolist()}")

        # Compute goal EE pose via FK
        goal_q = torch.tensor(goal_config, device=ta.device, dtype=ta.dtype)
        goal_pos, goal_quat = compute_fk(motion_gen, goal_q)
        print(f"  [GOAL {goal_name}]: Goal config: {goal_config}")
        print(f"  [GOAL {goal_name}]: Goal EE (base frame): pos={goal_pos.tolist()}, "
              f"quat={goal_quat.tolist()}")

        # Plan
        result = plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat)
        print(f"  [GOAL {goal_name}]: {result}")

        if not result.success:
            print(f"  [GOAL {goal_name}] FAIL: Planning failed — {result.status}")
            goal_results[goal_name] = False
            continue

        # Execute trajectory — position control only
        print(f"  [GOAL {goal_name}]: Executing {result.num_waypoints} waypoints "
              f"({steps_per_waypoint} sim steps each)...")

        for wp_idx in range(len(result.positions)):
            waypoint = result.positions[wp_idx]

            # NaN safety check
            if torch.isnan(robot.data.joint_pos[0, :]).any():
                print(f"  [GOAL {goal_name}] FAIL: NaN in joint positions at wp {wp_idx}")
                goal_results[goal_name] = False
                break

            # Position target: set liftkit + arm joints
            pos_target = robot.data.joint_pos[0:1, :].clone()
            for ci, si in enumerate(arm_liftkit_sim_ids):
                pos_target[0, si] = waypoint[ci].to(device)
            # Enforce liftkit mimic
            pos_target[0, liftkit_top_idx] = pos_target[0, liftkit_mid_idx]

            robot.set_joint_position_target(pos_target)

            # Step sim and update markers
            sim_dt = sim.get_physics_dt()
            num_envs = robot.data.root_pose_w.shape[0]
            for _ in range(steps_per_waypoint):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)

                # Update EE marker (world frame)
                tool0_pos_w, tool0_quat_w = get_all_tool0_poses_w(robot)
                ee_marker.visualize(tool0_pos_w, tool0_quat_w)

                # Update goal marker: transform base-frame goal to world frame
                base_pos_w, base_quat_w = get_all_base_frames_w(robot)
                goal_pos_exp = goal_pos.unsqueeze(0).expand(num_envs, -1).to(base_pos_w.device)
                goal_quat_exp = goal_quat.unsqueeze(0).expand(num_envs, -1).to(base_quat_w.device)
                goal_pos_w, goal_quat_w = combine_frame_transforms(
                    base_pos_w, base_quat_w, goal_pos_exp, goal_quat_exp,
                )
                goal_marker.visualize(goal_pos_w, goal_quat_w)

            # Progress logging
            if (wp_idx + 1) % 50 == 0 or wp_idx == len(result.positions) - 1:
                sim_tool0_base_pos, _ = get_tool0_pose_in_base(robot)
                track_err = torch.norm(
                    sim_tool0_base_pos[0].to(goal_pos.device) - goal_pos
                ).item()
                print(f"  [GOAL {goal_name}]: wp {wp_idx + 1}/{len(result.positions)}, "
                      f"track_err: {track_err * 1000:.1f} mm")
        else:
            # Only reach here if loop completed without break
            # Hold 200 steps to let PD controller settle
            print(f"  [GOAL {goal_name}]: Trajectory done. Holding 200 steps...")
            sim_step_n(sim, scene, 200)

            # Final verification in base frame
            sim_tool0_base_pos, sim_tool0_base_quat = get_tool0_pose_in_base(robot)

            pos_err = torch.norm(
                sim_tool0_base_pos[0].to(goal_pos.device) - goal_pos
            ).item()
            quat_err = quat_angular_error(
                sim_tool0_base_quat[0].to(goal_quat.device), goal_quat
            ).item()

            print(f"  [GOAL {goal_name}]: Final EE  (base): pos={sim_tool0_base_pos[0].tolist()}")
            print(f"  [GOAL {goal_name}]: Goal      (base): pos={goal_pos.tolist()}")
            print(f"  [GOAL {goal_name}]: Pos error: {pos_err * 1000:.1f} mm, "
                  f"Quat error: {math.degrees(quat_err):.2f} deg")

            pos_ok = pos_err < 0.020
            quat_ok = quat_err < math.radians(10.0)

            if pos_ok and quat_ok:
                print(f"  [GOAL {goal_name}] PASS: EE reached goal "
                      f"(pos: {pos_err * 1000:.1f} mm, quat: {math.degrees(quat_err):.2f} deg)")
                goal_results[goal_name] = True
            else:
                reasons = []
                if not pos_ok:
                    reasons.append(f"pos {pos_err * 1000:.1f}mm > 20mm")
                if not quat_ok:
                    reasons.append(f"quat {math.degrees(quat_err):.2f}deg > 10deg")
                print(f"  [GOAL {goal_name}] FAIL: {', '.join(reasons)}")
                goal_results[goal_name] = False

    all_ok = all(goal_results.values()) and len(goal_results) == len(GOAL_CONFIGS)
    passed = sum(1 for v in goal_results.values() if v)
    print(f"\nSTEP 3 {'PASS' if all_ok else 'FAIL'}: "
          f"{passed}/{len(GOAL_CONFIGS)} goals reached (pos AND quat verified)")
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

    # Steps 2-3 need cuRobo
    motion_gen = None
    tensor_args = None
    if step in (0, 2, 3):
        motion_gen, tensor_args = build_motion_gen()

    # Step 2: FK alignment
    if step in (0, 2):
        results[2] = step2_fk_alignment(sim, scene, motion_gen, tensor_args)
        if step == 2:
            print(f"\n{'=' * 70}")
            print(f"SUMMARY: Step 2 {'PASS' if results[2] else 'FAIL'}")
            print(f"{'=' * 70}")
            return

    # Step 3: Plan + execute all 4 goals
    if step in (0, 3):
        ok, goal_results = step3_plan_and_execute(sim, scene, motion_gen, tensor_args)
        results[3] = ok

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for s in sorted(results):
        status = "PASS" if results[s] else "FAIL"
        print(f"  Step {s}: {status}")
    if 3 in results:
        for name in GOAL_NAMES:
            tag = "PASS" if goal_results.get(name, False) else "FAIL"
            print(f"    Goal {name}: {tag}")
    all_pass = all(results.values())
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
    simulation_app.close()
