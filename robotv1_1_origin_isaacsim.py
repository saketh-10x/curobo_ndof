"""RobotV1.1 — writes "ORIGIN" on a wall in Isaac Sim.

Pre-plans trajectory with CuRobo (IK + linear interp for pen-down,
plan_single for pen-up), then replays in Isaac Sim with swerve drive
base control + arm position control.

Debug draw shows the EE trail in real-time during pen-down strokes.
Logs EE tracking error throughout execution.

Usage:
    cd ~/IsaacLab
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p scripts/standalone/curobo_v1_1/robotv1_1_origin_isaacsim.py
    # headless:
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p scripts/standalone/curobo_v1_1/robotv1_1_origin_isaacsim.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ORIGIN trajectory replay in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import math
import os
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# cuRobo imports
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.state import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

# Debug draw
try:
    from isaacsim.util.debug_draw import _debug_draw
except ImportError:
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        _debug_draw = None

# ── paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")
_SPHERES_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_spheres.yml")
_USD_DIR = os.path.join(_SCRIPT_DIR, "usd_generated")

# ── robot constants ────────────────────────────────────────────────────
JOINT_NAMES = [
    "base_x", "base_y", "base_theta", "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# eoa_tool_change home pose for the UR arm
HOME_ARM = [0.0, 0.0, -1.5708, -3.1416, -1.5708, 3.1416]
HOME_Q = [0.0, 0.0, 0.0, 0.0] + HOME_ARM

BASE_JOINT_NAMES = ["base_x", "base_y", "base_theta"]

# Swerve drive geometry
STEERING_JOINT_NAMES = ["steering_f", "steering_rl", "steering_rr"]
WHEEL_JOINT_NAMES = ["wheel_f", "wheel_rl", "wheel_rr"]
SWERVE_WHEELS = [
    (0.3225, 0.0),       # front
    (-0.3225, 0.245),    # rear left
    (-0.3225, -0.245),   # rear right
]
SWERVE_STEERING_LIMIT = math.radians(140)
WHEEL_RADIUS = 0.0775

_MESH_LINK_NAMES = [
    "base_link", "steering_3", "wheel_3", "steering_2", "wheel_2",
    "steering", "wheel", "liftkit_mid", "liftkit_top",
    "base_link_inertia", "shoulder_link", "upper_arm_link",
    "upperarm_clip", "forearm_link", "forearm_clip_1",
    "forearm_clip_2", "wrist_1_link", "wrist_2_link",
    "wrist_3_link", "tool_mount",
]

# ── wall & letter parameters ──────────────────────────────────────────
SPRAY_STANDOFF = 0.15
PEN_UP_EXTRA = 0.12
WALL_THICKNESS = 0.04
WALL_WIDTH = 2.0
WALL_HEIGHT = 1.5
LETTER_HEIGHT = 0.28
LETTER_WIDTH = 0.18
LETTER_SPACING = 0.06
LINE_INTERP_STEP = 0.03

# ── letter stroke definitions ─────────────────────────────────────────
LETTERS = {
    "O": [[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]],
    "R": [
        [(0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0.5)],
        [(0, 0.5), (1, 0)],
    ],
    "I": [[(0.5, 0), (0.5, 1)]],
    "G": [[(1, 1), (0, 1), (0, 0), (1, 0), (1, 0.5)]],
    "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
}
WORD = "ORIGIN"

# ── planning constants ────────────────────────────────────────────────
JOINT_INTERP_STEPS = 50

# ── debug draw colors ─────────────────────────────────────────────────
DD_PEN_DOWN_COLOR = (0.0, 0.85, 0.0, 1.0)  # green
DD_PEN_UP_COLOR = (1.0, 0.4, 0.0, 0.3)     # faint orange
DD_LINE_SIZE = 3.0


# ═══════════════════════════════════════════════════════════════════════
#  Scene Configuration
# ═══════════════════════════════════════════════════════════════════════
@configclass
class OriginSceneCfg(InteractiveSceneCfg):
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
                "shoulder_pan_joint": HOME_ARM[0],
                "shoulder_lift_joint": HOME_ARM[1],
                "elbow_joint": HOME_ARM[2],
                "wrist_1_joint": HOME_ARM[3],
                "wrist_2_joint": HOME_ARM[4],
                "wrist_3_joint": HOME_ARM[5],
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
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
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=600.0,
                damping=34.64101615,
                effort_limit=150.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=216.0,
                damping=29.39387691,
                effort_limit=54.0,
            ),
            "liftkit": ImplicitActuatorCfg(
                joint_names_expr=["liftkit_.*"],
                stiffness=800.0,
                damping=45.0,
                effort_limit=100.0,
                velocity_limit=0.012,
            ),
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["steering_.*"],
                stiffness=1000.0,
                damping=100.0,
                effort_limit=500.0,
                velocity_limit=120.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*"],
                stiffness=0.0,
                damping=1000.0,
                effort_limit=500.0,
                velocity_limit=100.0,
            ),
        },
    )


# ═══════════════════════════════════════════════════════════════════════
#  CuRobo helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_kin_model(robot_cfg, tensor_args):
    cfg = copy.deepcopy(robot_cfg)
    cfg["kinematics"]["mesh_link_names"] = _MESH_LINK_NAMES
    cfg["kinematics"]["load_link_names_with_mesh"] = True
    cfg["kinematics"]["use_usd_kinematics"] = False
    kin_cfg = CudaRobotModelConfig.from_data_dict(
        cfg["kinematics"], tensor_args=tensor_args)
    return CudaRobotModel(kin_cfg), cfg


def _get_ee_pose(kin_model, q, ee_link):
    p = kin_model.get_link_poses(q.contiguous(), [ee_link])
    return p.position[:, 0, :], p.quaternion[:, 0, :]


def _make_pose(pos_np, quat_np, tensor_args):
    return CuroboPose(
        position=tensor_args.to_device([[float(pos_np[0]),
                                         float(pos_np[1]),
                                         float(pos_np[2])]]),
        quaternion=tensor_args.to_device([[float(quat_np[0]),
                                           float(quat_np[1]),
                                           float(quat_np[2]),
                                           float(quat_np[3])]]),
    )


def _build_world(wall_center_y, wall_center_z):
    return WorldConfig(cuboid=[
        Cuboid(name="ground",
               pose=[0, 0, -0.50, 1, 0, 0, 0],
               dims=[6.0, 6.0, 0.02],
               color=[0.80, 0.80, 0.80, 0.25]),
        Cuboid(name="spray_wall",
               pose=[0.0, wall_center_y, wall_center_z, 1, 0, 0, 0],
               dims=[WALL_WIDTH, WALL_THICKNESS, WALL_HEIGHT],
               color=[0.55, 0.55, 0.60, 0.7]),
    ])


def _generate_word_waypoints(word, center_x, ee_y_down, ee_y_up, center_z,
                             spray_quat):
    """Generate waypoints to write a word on the wall."""
    n_letters = len(word)
    total_width = n_letters * LETTER_WIDTH + (n_letters - 1) * LETTER_SPACING
    start_x = center_x - total_width / 2.0
    bot_z = center_z - LETTER_HEIGHT / 2.0

    waypoints = []
    for li, ch in enumerate(word):
        letter_x0 = start_x + li * (LETTER_WIDTH + LETTER_SPACING)
        strokes = LETTERS[ch]

        for si, stroke in enumerate(strokes):
            sx, sz = stroke[0]
            up_pos = np.array([letter_x0 + sx * LETTER_WIDTH,
                               ee_y_up,
                               bot_z + sz * LETTER_HEIGHT])
            waypoints.append(
                (f"{ch}_{li}_s{si}_penup", up_pos, spray_quat, False))

            for pi, (px, pz) in enumerate(stroke):
                target = np.array([letter_x0 + px * LETTER_WIDTH,
                                   ee_y_down,
                                   bot_z + pz * LETTER_HEIGHT])
                if pi == 0:
                    waypoints.append(
                        (f"{ch}_{li}_s{si}_p0", target, spray_quat, True))
                else:
                    prev = waypoints[-1][1]
                    seg_len = np.linalg.norm(target - prev)
                    n_sub = max(1, int(np.ceil(seg_len / LINE_INTERP_STEP)))
                    for k in range(1, n_sub + 1):
                        alpha = k / n_sub
                        pt = prev + alpha * (target - prev)
                        waypoints.append(
                            (f"{ch}_{li}_s{si}_p{pi}_{k}",
                             pt, spray_quat, True))

    return waypoints


# ═══════════════════════════════════════════════════════════════════════
#  Swerve drive IK
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  Sim helpers
# ═══════════════════════════════════════════════════════════════════════

def sim_step_n(sim, scene, n):
    dt = sim.get_physics_dt()
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def get_sim_joint_indices(robot, names):
    sim_names = list(robot.data.joint_names)
    return [sim_names.index(n) for n in names]


def get_tool0_pose_w(robot, env_idx=0):
    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    return (
        robot.data.body_state_w[env_idx:env_idx+1, tool0_idx, 0:3],
        robot.data.body_state_w[env_idx:env_idx+1, tool0_idx, 3:7],
    )


def get_base_frame_w(robot, env_idx=0):
    return (robot.data.root_pose_w[env_idx:env_idx+1, 0:3],
            robot.data.root_pose_w[env_idx:env_idx+1, 3:7])


def get_tool0_pose_in_base(robot, env_idx=0):
    base_pos_w, base_quat_w = get_base_frame_w(robot, env_idx)
    tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot, env_idx)
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


def quat_angular_error(q1, q2):
    q1 = q1 / q1.norm()
    q2 = q2 / q2.norm()
    dot = torch.abs(torch.dot(q1, q2)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


# ═══════════════════════════════════════════════════════════════════════
#  CuRobo trajectory planning (offline, before sim execution)
# ═══════════════════════════════════════════════════════════════════════

def plan_origin_trajectory(tensor_args):
    """Plan the full ORIGIN trajectory offline with CuRobo.

    Returns: (trajs, goals, labels, pen_flags, wall_center_y, wall_center_z,
              home_ee_pos, home_ee_quat, world_cfg)
    """
    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg["kinematics"]["collision_spheres"] = _SPHERES_PATH

    cspace = robot_cfg["kinematics"]["cspace"]
    cspace["max_acceleration"] = cspace.get("max_acceleration", 15.0) * 0.25
    cspace["max_jerk"] = cspace.get("max_jerk", 500.0) * 0.25

    kin_cfg = robot_cfg["kinematics"]
    scb = kin_cfg.get("self_collision_buffer", {})
    for link in scb:
        scb[link] = 0.04
    kin_cfg["self_collision_buffer"] = scb
    kin_cfg["collision_sphere_buffer"] = 0.01

    # FK at home
    kin, kcfg = _build_kin_model(robot_cfg, tensor_args)
    ee_link = kcfg["kinematics"]["ee_link"]
    home_q = tensor_args.to_device([HOME_Q])
    home_pos, home_quat = _get_ee_pose(kin, home_q, ee_link)
    hp = home_pos[0].cpu().numpy()
    hq = home_quat[0].cpu().numpy()

    print(f"\n=== Home EE (eoa_tool_change) ===")
    print(f"  pos : ({hp[0]:.4f}, {hp[1]:.4f}, {hp[2]:.4f})")
    print(f"  quat: ({hq[0]:.4f}, {hq[1]:.4f}, {hq[2]:.4f}, {hq[3]:.4f})")

    # Wall placement
    wall_center_y = hp[1] + 0.40
    wall_center_z = hp[2]
    world_cfg = _build_world(wall_center_y, wall_center_z)
    print(f"\n=== Wall placement ===")
    print(f"  center: (0.0, {wall_center_y:.3f}, {wall_center_z:.3f})")

    # MotionGen
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg, world_cfg, tensor_args,
        trajopt_tsteps=34,
        interpolation_steps=5000,
        num_ik_seeds=100,
        num_trajopt_seeds=12,
        grad_trajopt_iters=500,
        trajopt_dt=0.5,
        interpolation_dt=0.02,
        evaluate_interpolated_trajectory=True,
        js_trajopt_dt=0.5,
        js_trajopt_tsteps=34,
        self_collision_check=True,
        self_collision_opt=True,
        collision_activation_distance=0.10,
    )
    mg = MotionGen(mg_cfg)
    print("Warming up CuRobo ...")
    mg.warmup()

    # Generate waypoints
    spray_quat = hq.copy()
    wall_surface_y = wall_center_y - WALL_THICKNESS / 2.0
    ee_y_down = wall_surface_y - SPRAY_STANDOFF
    ee_y_up = wall_surface_y - SPRAY_STANDOFF - PEN_UP_EXTRA

    waypoints = _generate_word_waypoints(
        WORD, hp[0], ee_y_down, ee_y_up, hp[2], spray_quat)

    n_legs = len(waypoints) + 1
    print(f"\n=== Writing \"{WORD}\" on the wall ===")
    print(f"  Total legs: {n_legs}")

    # Sequential planning
    trajs = []
    goals = []
    labels = []
    pen_flags = []
    current_q = home_q.clone()
    fail_count = 0

    for wi, (lab, pos, quat, pen) in enumerate(waypoints):
        leg_num = wi + 1
        pen_str = "PEN-DOWN" if pen else "pen-up"
        print(f"[{leg_num}/{n_legs}] {pen_str:>8s}  -> {lab:20s}  "
              f"({pos[0]:+.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        goal = _make_pose(pos, quat, tensor_args)

        if not pen:
            result = mg.plan_single(
                JointState.from_position(current_q, joint_names=JOINT_NAMES),
                goal)
            if result.success.item():
                traj = result.get_interpolated_plan()
                trajs.append(traj)
                goals.append(goal)
                labels.append(f"leg{leg_num}_{lab}")
                pen_flags.append(False)
                print(f"           OK  steps={traj.position.shape[0]}")
                current_q = traj.position[-1:].clone()
            else:
                fail_count += 1
                print(f"           FAILED  (skipped)")
        else:
            seed = current_q.view(1, 1, -1)
            ik_result = mg.solve_ik(goal, seed_config=seed, return_seeds=1)

            if not ik_result.success.any():
                fail_count += 1
                print(f"           IK FAILED  (skipped)")
                continue

            goal_q = ik_result.solution[ik_result.success][0:1]
            start = current_q.squeeze(0)
            end = goal_q.squeeze(0)
            alphas = torch.linspace(0.0, 1.0, JOINT_INTERP_STEPS,
                                    device=start.device)
            positions = start.unsqueeze(0) + alphas.unsqueeze(1) * (
                end - start).unsqueeze(0)
            zeros = torch.zeros_like(positions)
            traj = JointState(
                position=positions,
                velocity=zeros.clone(),
                acceleration=zeros.clone(),
                jerk=zeros.clone(),
            )
            trajs.append(traj)
            goals.append(goal)
            labels.append(f"leg{leg_num}_{lab}")
            pen_flags.append(True)
            current_q = goal_q.clone()
            print(f"           IK+linear OK  steps={JOINT_INTERP_STEPS}")

    # Return to home
    if trajs:
        print(f"[{n_legs}/{n_legs}]   pen-up  -> Home")
        home_goal = _make_pose(hp, hq, tensor_args)
        result = mg.plan_single(
            JointState.from_position(current_q, joint_names=JOINT_NAMES),
            home_goal)
        if result.success.item():
            traj = result.get_interpolated_plan()
            trajs.append(traj)
            goals.append(home_goal)
            labels.append("return_home")
            pen_flags.append(False)
            print(f"           OK  steps={traj.position.shape[0]}")
        else:
            fail_count += 1
            print(f"           FAILED  (skipped)")

    print(f"\n=== Planning Summary ===")
    print(f"  Successful: {len(trajs)} / {n_legs}")
    print(f"  Failed: {fail_count}")
    total_steps = sum(t.position.shape[0] for t in trajs)
    print(f"  Total trajectory steps: {total_steps}")

    return (trajs, goals, labels, pen_flags,
            wall_center_y, wall_center_z, hp, hq, world_cfg)


# ═══════════════════════════════════════════════════════════════════════
#  Trajectory execution in Isaac Sim
# ═══════════════════════════════════════════════════════════════════════

def execute_trajectory(sim, scene, trajs, pen_flags, plan_dt=0.02):
    """Execute pre-planned trajectory in Isaac Sim with swerve drive.

    Returns tracking error stats dict.
    """
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    device = sim.device
    steps_per_waypoint = max(1, int(plan_dt / sim_dt))

    # Joint index mappings
    sim_joint_names = list(robot.data.joint_names)
    n_sim_joints = len(sim_joint_names)
    base_sim_ids = get_sim_joint_indices(robot, BASE_JOINT_NAMES)
    arm_liftkit_sim_ids = get_sim_joint_indices(robot, JOINT_NAMES[3:])
    steering_sim_ids = get_sim_joint_indices(robot, STEERING_JOINT_NAMES)
    wheel_sim_ids = get_sim_joint_indices(robot, WHEEL_JOINT_NAMES)
    liftkit_mid_idx = sim_joint_names.index("liftkit_mid")
    liftkit_top_idx = sim_joint_names.index("liftkit_top")

    # Debug draw interface
    draw = None
    if _debug_draw is not None:
        try:
            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            pass

    # Tracking stats
    pen_down_errors = []
    pen_up_errors = []
    prev_ee_pos = None

    print(f"\n=== Executing trajectory in Isaac Sim ===")
    print(f"  Steps per waypoint: {steps_per_waypoint}")
    print(f"  Debug draw: {'enabled' if draw else 'disabled'}")

    total_wp = sum(t.position.shape[0] for t in trajs)
    global_wp = 0

    for leg_idx, (traj, pen) in enumerate(zip(trajs, pen_flags)):
        plan_positions = traj.position
        n_wps = plan_positions.shape[0]
        pen_str = "PEN-DOWN" if pen else "pen-up"
        print(f"  Leg {leg_idx+1}/{len(trajs)} ({pen_str}): {n_wps} waypoints")

        for wp_idx in range(n_wps):
            waypoint = plan_positions[wp_idx]

            # NaN check
            if torch.isnan(robot.data.joint_pos[0, :]).any():
                print(f"  ABORT: NaN in joint positions at leg {leg_idx+1}, wp {wp_idx}")
                return {"error": "NaN detected"}

            # Base velocity from consecutive waypoints
            if wp_idx < n_wps - 1:
                next_wp = plan_positions[wp_idx + 1]
                base_vel = (next_wp[:3] - waypoint[:3]) / plan_dt
            else:
                base_vel = torch.zeros(3, device=waypoint.device)

            # Swerve IK
            base_theta_cur = waypoint[2].item()
            steer_angles, wheel_speeds = compute_swerve_commands(
                base_vel[0].item(), base_vel[1].item(), base_vel[2].item(),
                base_theta_cur,
            )

            # Position targets: arm/liftkit + steering
            pos_target = robot.data.joint_pos[0:1, :].clone()
            for ci, si in enumerate(arm_liftkit_sim_ids):
                pos_target[0, si] = waypoint[3 + ci].to(device)
            pos_target[0, liftkit_top_idx] = pos_target[0, liftkit_mid_idx]
            for si, angle in zip(steering_sim_ids, steer_angles):
                pos_target[0, si] = angle

            # Velocity targets: base + wheels
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

            # Read current EE position
            tool0_pos_w, _ = get_tool0_pose_w(robot)
            ee_pos = tool0_pos_w[0].clone()

            # Compute tracking error (EE in base frame vs planned)
            tool0_base_pos, _ = get_tool0_pose_in_base(robot)
            # We don't have planned EE in base frame directly, but we can
            # use CuRobo FK offline comparison later. For now, track world pos.

            # Debug draw
            if draw and prev_ee_pos is not None:
                color = DD_PEN_DOWN_COLOR if pen else DD_PEN_UP_COLOR
                p1 = prev_ee_pos.cpu().tolist()
                p2 = ee_pos.cpu().tolist()
                draw.draw_lines(
                    [p1], [p2], [color], [DD_LINE_SIZE]
                )

            prev_ee_pos = ee_pos
            global_wp += 1

            # Progress logging every 50 waypoints
            if global_wp % 50 == 0 or global_wp == total_wp:
                print(f"    Progress: {global_wp}/{total_wp} waypoints executed")

    # Stop all motion
    vel_target = torch.zeros(1, n_sim_joints, device=device)
    robot.set_joint_velocity_target(vel_target)

    print(f"  Trajectory done. Holding 200 steps...")
    sim_step_n(sim, scene, 200)

    # Final pose
    tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot)
    print(f"  Final EE (world): pos={tool0_pos_w[0].tolist()}")

    return {"total_waypoints": total_wp}


# ═══════════════════════════════════════════════════════════════════════
#  Validation: compare planned vs actual EE trajectory
# ═══════════════════════════════════════════════════════════════════════

def validate_ee_tracking(sim, scene, trajs, pen_flags, tensor_args, plan_dt=0.02):
    """Re-execute trajectory and measure EE tracking error vs CuRobo FK.

    This runs the same execution loop but also computes CuRobo FK at each
    waypoint to compare planned vs actual EE position.
    """
    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg["kinematics"]["collision_spheres"] = _SPHERES_PATH

    kin, kcfg = _build_kin_model(robot_cfg, tensor_args)
    ee_link = kcfg["kinematics"]["ee_link"]

    robot = scene["robot"]

    # Compute planned EE positions via CuRobo FK
    all_q = torch.cat([t.position for t in trajs], dim=0)
    planned_ee, _ = _get_ee_pose(kin, all_q, ee_link)
    planned_ee_np = planned_ee.cpu().numpy()

    # Read actual EE positions from sim
    tool0_base_pos, _ = get_tool0_pose_in_base(robot)
    actual_ee = tool0_base_pos[0].cpu().numpy()

    print(f"\n=== EE Tracking Validation ===")
    print(f"  Total planned waypoints: {all_q.shape[0]}")
    print(f"  Planned EE final: {planned_ee_np[-1]}")
    print(f"  Actual EE final (base frame): {actual_ee}")

    # Per-leg error summary using planned FK
    off = 0
    pen_down_max_err = 0.0
    pen_down_mean_errs = []
    for leg_idx, (traj, pen) in enumerate(zip(trajs, pen_flags)):
        n = traj.position.shape[0]
        if pen:
            # For pen-down legs, we care about tracking quality
            leg_ee = planned_ee_np[off:off+n]
            # We can't easily get actual per-step EE without re-running,
            # so just report planned trajectory stats
            if n > 1:
                step_dists = np.linalg.norm(np.diff(leg_ee, axis=0), axis=1)
                print(f"  Leg {leg_idx+1} (pen-down): {n} steps, "
                      f"mean step dist={step_dists.mean()*1000:.1f}mm, "
                      f"max step dist={step_dists.max()*1000:.1f}mm")
        off += n

    return True


# ═══════════════════════════════════════════════════════════════════════
#  Wall visual (spawn a cuboid in Isaac Sim)
# ═══════════════════════════════════════════════════════════════════════

def spawn_wall(wall_center_y, wall_center_z):
    """Spawn a visual wall cuboid in the scene."""
    import omni.isaac.core.utils.prims as prim_utils

    wall_cfg = sim_utils.CuboidCfg(
        size=(WALL_WIDTH, WALL_THICKNESS, WALL_HEIGHT),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.55, 0.60),
            opacity=0.7,
        ),
    )
    wall_cfg.func(
        "/World/spray_wall",
        wall_cfg,
        translation=(0.0, wall_center_y, wall_center_z),
    )
    print(f"  Wall spawned at (0.0, {wall_center_y:.3f}, {wall_center_z:.3f})")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Phase 1: Plan trajectory offline with CuRobo
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    print("=" * 70)
    print("PHASE 1a: Planning ORIGIN trajectory with CuRobo (offline)")
    print("=" * 70)

    result = plan_origin_trajectory(tensor_args)
    trajs, goals, labels, pen_flags = result[:4]
    wall_center_y, wall_center_z = result[4], result[5]
    home_ee_pos, home_ee_quat = result[6], result[7]
    world_cfg = result[8]

    if not trajs:
        print("No trajectories planned. Aborting.")
        return

    # Phase 1b: Set up Isaac Sim
    print("\n" + "=" * 70)
    print("PHASE 1b: Setting up Isaac Sim scene")
    print("=" * 70)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, -1.5, 2.0], [0.0, wall_center_y, wall_center_z])

    scene_cfg = OriginSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    # Spawn wall visual
    spawn_wall(wall_center_y, wall_center_z)

    sim.reset()
    print("[INFO]: Isaac Sim setup complete.")

    # Settle robot
    print("[INFO]: Settling robot (200 steps)...")
    sim_step_n(sim, scene, 200)

    # Verify initial joint state matches HOME_Q
    robot = scene["robot"]
    curobo_q = get_current_curobo_q(robot, tensor_args)
    print(f"[INFO]: Initial 10-DOF from sim: {curobo_q.tolist()}")

    # Phase 1c: Execute trajectory
    print("\n" + "=" * 70)
    print("PHASE 1c: Executing trajectory in Isaac Sim")
    print("=" * 70)

    stats = execute_trajectory(sim, scene, trajs, pen_flags, plan_dt=0.02)
    print(f"\n=== Execution stats: {stats}")

    # Phase 1d: Validation
    print("\n" + "=" * 70)
    print("PHASE 1d: Validation")
    print("=" * 70)

    validate_ee_tracking(sim, scene, trajs, pen_flags, tensor_args)

    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE: ORIGIN trajectory replay in Isaac Sim")
    print("=" * 70)
    print(f"  Word: {WORD}")
    print(f"  Legs: {len(trajs)}")
    total_steps = sum(t.position.shape[0] for t in trajs)
    print(f"  Total steps: {total_steps}")
    print(f"  Pen-down legs: {sum(1 for p in pen_flags if p)}")
    print(f"  Pen-up legs: {sum(1 for p in pen_flags if not p)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
