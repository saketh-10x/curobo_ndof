"""cuRobo 10-DOF collision avoidance demo for V1.1.1 robot.

Spawns cuboid obstacles in the arm workspace and uses cuRobo's WorldConfig
to plan collision-free trajectories around them. Self-collision is also enabled
using sphere-based checking from the auto-generated XRDF.

3 test steps:
  Step 1: Spawn robot + obstacles, settle
  Step 2: Plan trajectories that must navigate around obstacles
  Step 3: Execute plans, verify EE reaches goals collision-free

Usage:
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p \
        scripts/standalone/curobo_v1_1/run_curobo_collision_avoidance.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cuRobo 10-DOF collision avoidance demo.")
parser.add_argument("--num_envs", type=int, default=1)
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
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
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
from curobo.geom.types import WorldConfig, Cuboid
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
JOINT_NAMES = [
    "base_x", "base_y", "base_theta",
    "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

BASE_JOINT_NAMES = ["base_x", "base_y", "base_theta"]
STEERING_JOINT_NAMES = ["steering_f", "steering_rl", "steering_rr"]
WHEEL_JOINT_NAMES = ["wheel_f", "wheel_rl", "wheel_rr"]
SWERVE_WHEELS = [
    (0.3225, 0.0),
    (-0.3225, 0.245),
    (-0.3225, -0.245),
]
SWERVE_STEERING_LIMIT = math.radians(140)
WHEEL_RADIUS = 0.0775

##
# World obstacles — cuRobo uses [x, y, z, qw, qx, qy, qz]
##
OBSTACLES = [
    Cuboid(
        name="ground",
        dims=[5.0, 5.0, 0.1],
        pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    ),
    Cuboid(
        name="obstacle_front",
        dims=[0.3, 0.3, 0.6],
        pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
    ),
    Cuboid(
        name="obstacle_left",
        dims=[0.2, 0.6, 0.4],
        pose=[0.0, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0],
    ),
]

# Goal configs that require planning around obstacles.
# These are chosen so a straight-line path would collide.
# 10-DOF: [base_x, base_y, base_theta, liftkit_mid, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
GOAL_CONFIGS = [
    # Goal A: reach past the front obstacle (arm must go around/over it)
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.5, -2.0, 0.0, 0.0],
    # Goal B: reach past the left obstacle (arm must go around it)
    [0.0, 0.0, 0.0, 0.0, 1.5708, -1.0, 1.0, -1.5708, 0.0, 0.0],
    # Goal C: return to retract (safe home position)
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
]
GOAL_NAMES = ["A_past_front", "B_past_left", "C_retract"]


##
# Result container
##
@dataclass
class PlanResult:
    success: bool
    joint_names: List[str]
    positions: Optional[torch.Tensor]
    velocities: Optional[torch.Tensor]
    accelerations: Optional[torch.Tensor]
    dt: float
    solve_time_s: float
    motion_time_s: float
    status: str = ""

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
# Scene Configuration — robot + obstacles
##
@configclass
class CollisionSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, light, V1.1.1 robot, and cuboid obstacles."""

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
                "base_x": 0.0, "base_y": 0.0, "base_theta": 0.0,
                "steering_f": 0.0, "steering_rl": 0.0, "steering_rr": 0.0,
                "wheel_f": 0.0, "wheel_rl": 0.0, "wheel_rr": 0.0,
                "liftkit_mid": 0.0, "liftkit_top": 0.0,
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
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_x", "base_y", "base_theta"],
                effort_limit=1e5, stiffness=0.0, damping=1e5,
            ),
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=20000.0, damping=2000.0, effort_limit=19800.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=20000.0, damping=2000.0, effort_limit=19800.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=20000.0, damping=2000.0, effort_limit=19800.0,
            ),
            "liftkit": ImplicitActuatorCfg(
                joint_names_expr=["liftkit_.*"],
                stiffness=8000000.0, damping=45000.0,
                effort_limit=70000.0, velocity_limit=0.012,
            ),
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["steering_.*"],
                stiffness=1000.0, damping=100.0,
                effort_limit=500.0, velocity_limit=120.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*"],
                stiffness=0.0, damping=1000.0,
                effort_limit=500.0, velocity_limit=100.0,
            ),
        },
    )

    # Obstacle 1: front box blocking forward reach
    obstacle_front = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleFront",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Obstacle 2: left box blocking left reach
    obstacle_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.6, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.5, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


##
# Planner helpers
##

def build_motion_gen():
    """Build cuRobo MotionGen with world obstacles and self-collision."""
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg["kinematics"]["collision_spheres"] = _SPHERES_PATH

    world_cfg = WorldConfig(cuboid=OBSTACLES)

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=world_cfg,
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_ik_seeds=30,
        num_graph_seeds=12,
        num_trajopt_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"obb": 20, "mesh": 10},
        trajopt_tsteps=32,
        collision_activation_distance=0.025,
        self_collision_check=True,
        position_threshold=0.005,
        rotation_threshold=0.05,
    )

    motion_gen = MotionGen(mg_cfg)

    print("[INFO]: Warming up cuRobo MotionGen (with world obstacles)...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[INFO]: cuRobo warmup complete.")

    return motion_gen, tensor_args


def compute_fk(motion_gen, q):
    """Compute FK for joint positions q. Returns (ee_pos [3], ee_quat [4])."""
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

    print(f"  cuRobo pos_err: {result.position_error}, rot_err: {result.rotation_error}")

    traj = result.get_interpolated_plan()
    if traj.joint_names != JOINT_NAMES:
        traj = traj.get_ordered_joint_state(JOINT_NAMES)

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
# Swerve drive IK
##

def swerve_ik_wheel(vx_body, vy_body, omega, wheel_x, wheel_y, limit):
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
        wheel_speed = speed / WHEEL_RADIUS
        if i == 0:
            wheel_speed = -wheel_speed
        wheel_speeds.append(wheel_speed)

    return steer_angles, wheel_speeds


##
# Simulation helpers
##

def sim_step_n(sim, scene, n):
    dt = sim.get_physics_dt()
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def get_sim_joint_indices(robot, names):
    sim_names = list(robot.data.joint_names)
    return [sim_names.index(n) for n in names]


def get_base_frame_w(robot):
    return robot.data.root_pose_w[0:1, 0:3], robot.data.root_pose_w[0:1, 3:7]


def get_tool0_pose_w(robot):
    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    return (
        robot.data.body_state_w[0:1, tool0_idx, 0:3],
        robot.data.body_state_w[0:1, tool0_idx, 3:7],
    )


def get_tool0_pose_in_base(robot):
    base_pos_w, base_quat_w = get_base_frame_w(robot)
    tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot)
    return subtract_frame_transforms(base_pos_w, base_quat_w, tool0_pos_w, tool0_quat_w)


def get_current_curobo_q(robot, tensor_args):
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
# Main steps
##

def step1_spawn_and_settle(sim, scene):
    """Step 1: Spawn robot + obstacles and settle."""
    print("\n" + "=" * 70)
    print("STEP 1: Spawn Robot + Obstacles, Settle")
    print("=" * 70)

    robot = scene["robot"]

    print("[STEP1]: Settling scene (200 steps)...")
    sim_step_n(sim, scene, 200)

    # Verify robot settled
    arm_names = ARM_JOINT_NAMES
    arm_indices = get_sim_joint_indices(robot, arm_names)
    arm_target = torch.tensor([0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
                              device=sim.device, dtype=torch.float32)
    arm_actual = robot.data.joint_pos[0, arm_indices]
    errors = (arm_actual - arm_target).abs()
    max_err = errors.max().item()

    print(f"[STEP1]: Arm joint errors: {errors.tolist()}")
    print(f"[STEP1]: Max arm error: {max_err:.4f} rad")

    # Verify obstacles exist
    try:
        obs_front = scene["obstacle_front"]
        obs_left = scene["obstacle_left"]
        print(f"[STEP1]: obstacle_front pos: {obs_front.data.root_pos_w[0].tolist()}")
        print(f"[STEP1]: obstacle_left pos: {obs_left.data.root_pos_w[0].tolist()}")
    except Exception as e:
        print(f"[STEP1]: WARNING: Could not read obstacle data: {e}")

    if max_err < 0.05:
        print("STEP 1 PASS: Robot settled, obstacles spawned")
        return True
    else:
        print(f"STEP 1 FAIL: Max arm error {max_err:.4f} rad exceeds 0.05 rad")
        return False


def step2_plan_around_obstacles(sim, scene, motion_gen, tensor_args):
    """Step 2: Plan trajectories around obstacles."""
    print("\n" + "=" * 70)
    print("STEP 2: Plan Trajectories Around Obstacles")
    print("=" * 70)

    robot = scene["robot"]
    ta = tensor_args

    plans = []
    all_ok = True

    for i, (cfg, name) in enumerate(zip(GOAL_CONFIGS, GOAL_NAMES)):
        print(f"\n--- Goal {name} ({i + 1}/{len(GOAL_CONFIGS)}) ---")

        current_q = get_current_curobo_q(robot, tensor_args)
        print(f"  [PLAN {name}]: Start 10-DOF: {current_q.tolist()}")

        goal_q = torch.tensor(cfg, device=ta.device, dtype=ta.dtype)
        goal_pos, goal_quat = compute_fk(motion_gen, goal_q)
        print(f"  [PLAN {name}]: Goal EE (base frame): pos={goal_pos.tolist()}, "
              f"quat={goal_quat.tolist()}")

        result = plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat)
        print(f"  [PLAN {name}]: {result}")

        if not result.success:
            print(f"  [PLAN {name}] FAIL: {result.status}")
            plans.append((False, None, None, None))
            all_ok = False
        else:
            final_pos, final_quat = compute_fk(motion_gen, result.positions[-1])
            pos_err = torch.norm(final_pos - goal_pos).item()
            quat_err = quat_angular_error(final_quat, goal_quat).item()
            print(f"  [PLAN {name}]: FK diagnostic — pos err: {pos_err * 1000:.1f} mm, "
                  f"quat err: {math.degrees(quat_err):.2f} deg")
            print(f"  [PLAN {name}] PASS: {result.num_waypoints} waypoints")
            plans.append((True, result, goal_pos, goal_quat))

    passed = sum(1 for ok, *_ in plans if ok)
    print(f"\nSTEP 2 {'PASS' if all_ok else 'FAIL'}: {passed}/{len(plans)} plans succeeded")
    return all_ok, plans


def step3_execute_and_verify(sim, scene, motion_gen, tensor_args, plans):
    """Step 3: Execute all plans and verify EE reaches goals."""
    print("\n" + "=" * 70)
    print("STEP 3: Execute Plans + Verify EE Reach")
    print("=" * 70)

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    steps_per_waypoint = 4
    device = sim.device

    sim_joint_names = list(robot.data.joint_names)
    n_sim_joints = len(sim_joint_names)
    base_sim_ids = get_sim_joint_indices(robot, BASE_JOINT_NAMES)
    arm_liftkit_sim_ids = get_sim_joint_indices(robot, JOINT_NAMES[3:])
    steering_sim_ids = get_sim_joint_indices(robot, STEERING_JOINT_NAMES)
    wheel_sim_ids = get_sim_joint_indices(robot, WHEEL_JOINT_NAMES)
    liftkit_mid_idx = sim_joint_names.index("liftkit_mid")
    liftkit_top_idx = sim_joint_names.index("liftkit_top")

    # Create markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    goal_frame_cfg = FRAME_MARKER_CFG.copy()
    goal_frame_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    goal_marker = VisualizationMarkers(goal_frame_cfg.replace(prim_path="/Visuals/ee_goal"))

    goal_results = {}

    for i, ((ok, plan, gpos, gquat), name) in enumerate(zip(plans, GOAL_NAMES)):
        print(f"\n{'~' * 50}")
        print(f"  Goal {name} ({i + 1}/{len(plans)})")
        print(f"{'~' * 50}")

        if not ok or plan is None:
            print(f"  [EXEC {name}] SKIP: No valid plan")
            goal_results[name] = False
            continue

        # Re-plan from current sim state
        print(f"  [EXEC {name}]: Re-planning from current sim state...")
        ta = tensor_args
        current_q = get_current_curobo_q(robot, tensor_args)
        goal_q = torch.tensor(GOAL_CONFIGS[i], device=ta.device, dtype=ta.dtype)
        re_gpos, re_gquat = compute_fk(motion_gen, goal_q)
        re_plan = plan_to_pose(motion_gen, tensor_args, current_q, re_gpos, re_gquat)

        if not re_plan.success:
            print(f"  [EXEC {name}] FAIL: Re-planning failed — {re_plan.status}")
            goal_results[name] = False
            continue

        plan_positions = re_plan.positions
        plan_dt = re_plan.dt
        print(f"  [EXEC {name}]: Executing {re_plan.num_waypoints} waypoints...")

        init_base_pos_w, init_base_quat_w = get_base_frame_w(robot)
        init_base_pos_w = init_base_pos_w.clone()
        init_base_quat_w = init_base_quat_w.clone()

        for wp_idx in range(len(plan_positions)):
            waypoint = plan_positions[wp_idx]

            if torch.isnan(robot.data.joint_pos[0, :]).any():
                print(f"  [EXEC {name}] FAIL: NaN at wp {wp_idx}")
                goal_results[name] = False
                break

            # Base velocity
            if wp_idx < len(plan_positions) - 1:
                next_wp = plan_positions[wp_idx + 1]
                base_vel = (next_wp[:3] - waypoint[:3]) / plan_dt
            else:
                base_vel = torch.zeros(3, device=waypoint.device)

            base_theta_cur = waypoint[2].item()
            steer_angles, wheel_speeds = compute_swerve_commands(
                base_vel[0].item(), base_vel[1].item(), base_vel[2].item(),
                base_theta_cur,
            )

            # Position targets
            pos_target = robot.data.joint_pos[0:1, :].clone()
            for ci, si in enumerate(arm_liftkit_sim_ids):
                pos_target[0, si] = waypoint[3 + ci].to(device)
            pos_target[0, liftkit_top_idx] = pos_target[0, liftkit_mid_idx]
            for si, angle in zip(steering_sim_ids, steer_angles):
                pos_target[0, si] = angle

            # Velocity targets
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

                goal_pos_w, goal_quat_w = combine_frame_transforms(
                    init_base_pos_w, init_base_quat_w,
                    re_gpos.unsqueeze(0).to(init_base_pos_w.device),
                    re_gquat.unsqueeze(0).to(init_base_quat_w.device),
                )
                tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot)
                ee_marker.visualize(tool0_pos_w, tool0_quat_w)
                goal_marker.visualize(goal_pos_w, goal_quat_w)

            if (wp_idx + 1) % 50 == 0 or wp_idx == len(plan_positions) - 1:
                sim_tool0_base_pos, _ = get_tool0_pose_in_base(robot)
                track_err = torch.norm(
                    sim_tool0_base_pos[0].to(re_gpos.device) - re_gpos
                ).item()
                print(f"  [EXEC {name}]: wp {wp_idx + 1}/{len(plan_positions)}, "
                      f"track_err: {track_err * 1000:.1f} mm")

        else:
            # Loop completed without break (no NaN)
            vel_target = torch.zeros(1, n_sim_joints, device=device)
            robot.set_joint_velocity_target(vel_target)

            print(f"  [EXEC {name}]: Holding 200 steps...")
            sim_step_n(sim, scene, 200)

            # Verify in base frame
            sim_tool0_base_pos, sim_tool0_base_quat = get_tool0_pose_in_base(robot)
            pos_err = torch.norm(
                sim_tool0_base_pos[0].to(re_gpos.device) - re_gpos
            ).item()
            quat_err = quat_angular_error(
                sim_tool0_base_quat[0].to(re_gquat.device), re_gquat
            ).item()

            print(f"  [EXEC {name}]: Final pos error: {pos_err * 1000:.1f} mm, "
                  f"quat error: {math.degrees(quat_err):.2f} deg")

            pos_ok = pos_err < 0.030
            quat_ok = quat_err < math.radians(15.0)

            if pos_ok and quat_ok:
                print(f"  [EXEC {name}] PASS")
                goal_results[name] = True
            else:
                reasons = []
                if not pos_ok:
                    reasons.append(f"pos {pos_err * 1000:.1f}mm > 30mm")
                if not quat_ok:
                    reasons.append(f"quat {math.degrees(quat_err):.2f}deg > 15deg")
                print(f"  [EXEC {name}] FAIL: {', '.join(reasons)}")
                goal_results[name] = False
            continue

        # NaN break path
        if name not in goal_results:
            goal_results[name] = False

    all_ok = all(goal_results.values()) and len(goal_results) == len(plans)
    passed = sum(1 for v in goal_results.values() if v)
    print(f"\nSTEP 3 {'PASS' if all_ok else 'FAIL'}: "
          f"{passed}/{len(plans)} goals reached collision-free")
    return all_ok, goal_results


##
# Main
##

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 1.0])

    scene_cfg = CollisionSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Scene setup complete — robot + obstacles spawned.")

    results = {}

    # Step 1
    results[1] = step1_spawn_and_settle(sim, scene)

    # Build planner with world obstacles
    motion_gen, tensor_args = build_motion_gen()

    # Step 2
    ok, plans = step2_plan_around_obstacles(sim, scene, motion_gen, tensor_args)
    results[2] = ok

    # Step 3
    ok, goal_results = step3_execute_and_verify(sim, scene, motion_gen, tensor_args, plans)
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
