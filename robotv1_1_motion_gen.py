"""RobotV1.1 spraying action — Z-letter pattern on a large wall.

The robot starts at the eoa_tool_change home pose, does FK to find the EE pose,
places a large wall in front of the sprayer, then plans a "Z" letter path:

    →→→→→→→→  (top stroke — sweep right)
           ↙  (diagonal — top-right to bottom-left)
    →→→→→→→→  (bottom stroke — sweep right)

Each waypoint is a Cartesian EE pose offset from the wall surface (spray
standoff distance ~0.15 m), with orientation pointing into the wall.

Self-collision and world-collision are both enabled.
The USD contains the robot animation, the wall, ground, waypoint markers,
and dense EE/base path traces so every motion is visible.

Usage:
    cd ~/IsaacLab/scripts/standalone/curobo_v1_1
    python robotv1_1_motion_gen.py
"""

import copy
import os
from datetime import datetime

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.types import Cuboid, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

try:
    from curobo.util.usd_helper import UsdHelper
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

# ── paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")
_SPHERES_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_spheres.yml")

# ── robot constants ────────────────────────────────────────────────────
JOINT_NAMES = [
    "base_x", "base_y", "base_theta", "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# eoa_tool_change home pose for the UR arm
HOME_ARM = [0.0, 0.0, -1.5708, -3.1416, -1.5708, 3.1416]
HOME_Q = [0.0, 0.0, 0.0, 0.0] + HOME_ARM

_MESH_LINK_NAMES = [
    "base_link", "steering_3", "wheel_3", "steering_2", "wheel_2",
    "steering", "wheel", "liftkit_mid", "liftkit_top",
    "base_link_inertia", "shoulder_link", "upper_arm_link",
    "upperarm_clip", "forearm_link", "forearm_clip_1",
    "forearm_clip_2", "wrist_1_link", "wrist_2_link",
    "wrist_3_link", "tool_mount",
]

# ── spray pattern parameters ──────────────────────────────────────────
SPRAY_STANDOFF = 0.015          # distance from wall surface to EE
WALL_THICKNESS = 0.04          # thin wall
WALL_WIDTH = 2.0               # wall X extent
WALL_HEIGHT = 1.5              # wall Z extent (taller wall)
SWEEP_HALF_WIDTH = 0.80        # half-width of Z letter in X (near wall edges)
Z_HEIGHT = 1.00                # vertical span of the Z letter (near wall top/bottom)

# ── colours ────────────────────────────────────────────────────────────
COL_HOME      = [1.0, 1.0, 1.0, 1.0]    # white
COL_WALL      = [0.55, 0.55, 0.60, 0.7] # grey wall
COL_GROUND    = [0.80, 0.80, 0.80, 0.25]
COL_WAYPOINTS = [
    [1.0, 0.3, 0.0, 1.0],   # orange
    [0.0, 0.85, 0.0, 1.0],  # green
    [0.0, 0.35, 1.0, 1.0],  # blue
    [0.85, 0.0, 0.85, 1.0], # magenta
    [1.0, 0.85, 0.0, 1.0],  # yellow
    [0.0, 0.85, 0.85, 1.0], # cyan
    [1.0, 0.0, 0.0, 1.0],   # red
]
COL_PATH = [
    [1.0, 0.5, 0.2, 0.8],
    [0.3, 0.95, 0.3, 0.8],
    [0.3, 0.55, 1.0, 0.8],
    [0.9, 0.3, 0.9, 0.8],
    [1.0, 0.9, 0.2, 0.8],
    [0.2, 0.9, 0.9, 0.8],
    [1.0, 0.3, 0.3, 0.8],
]


# ═══════════════════════════════════════════════════════════════════════
#  World
# ═══════════════════════════════════════════════════════════════════════
def _build_world(wall_center_y, wall_center_z):
    """Ground plane + large wall positioned in front of the sprayer."""
    return WorldConfig(cuboid=[
        Cuboid(name="ground",
               pose=[0, 0, -0.50, 1, 0, 0, 0],
               dims=[6.0, 6.0, 0.02],
               color=COL_GROUND),
        Cuboid(name="spray_wall",
               pose=[0.0, wall_center_y, wall_center_z, 1, 0, 0, 0],
               dims=[WALL_WIDTH, WALL_THICKNESS, WALL_HEIGHT],
               color=COL_WALL),
    ])


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
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
    return Pose(
        position=tensor_args.to_device([[float(pos_np[0]),
                                         float(pos_np[1]),
                                         float(pos_np[2])]]),
        quaternion=tensor_args.to_device([[float(quat_np[0]),
                                           float(quat_np[1]),
                                           float(quat_np[2]),
                                           float(quat_np[3])]]),
    )


def plot_traj(trajectory, dt, joint_names, title="", file_name="traj.png"):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(14, 10))
    q   = trajectory.position.cpu().numpy()
    qd  = trajectory.velocity.cpu().numpy()
    qdd = trajectory.acceleration.cpu().numpy()
    qddd = trajectory.jerk.cpu().numpy()
    t = [i * dt for i in range(q.shape[0])]
    labels = joint_names if len(joint_names) == q.shape[-1] \
        else [str(i) for i in range(q.shape[-1])]
    for i in range(q.shape[-1]):
        axs[0].plot(t, q[:, i],    label=labels[i])
        axs[1].plot(t, qd[:, i],   label=labels[i])
        axs[2].plot(t, qdd[:, i],  label=labels[i])
        axs[3].plot(t, qddd[:, i], label=labels[i])
    axs[0].set_ylabel("Position (rad)")
    axs[1].set_ylabel("Velocity (rad/s)")
    axs[2].set_ylabel("Acceleration")
    axs[3].set_ylabel("Jerk")
    axs[3].set_xlabel("Time (s)")
    fig.suptitle(title or "Trajectory")
    axs[0].legend(fontsize=6, ncol=5, loc="upper right")
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close()
    print(f"  Plot saved: {file_name}")


# ═══════════════════════════════════════════════════════════════════════
#  USD export
# ═══════════════════════════════════════════════════════════════════════
def _save_usd(robot_cfg, trajs, goals, labels, home_ee_np,
              world_cfg, dt, save_path, base_frame, tensor_args):
    kin_model, usd_cfg = _build_kin_model(robot_cfg, tensor_args)
    ee_link = usd_cfg["kinematics"]["ee_link"]

    all_q = torch.cat([t.position for t in trajs], dim=0)

    # robot meshes
    meshes  = kin_model.get_robot_link_meshes()
    offsets = [m.pose for m in meshes]
    robot_world = WorldConfig(mesh=meshes)
    anim_links  = kin_model.kinematics_config.mesh_link_names
    anim_poses  = kin_model.get_link_poses(all_q.contiguous(), anim_links)
    for i, ival in enumerate(offsets):
        op = Pose.from_list(ival)
        np_ = Pose(anim_poses.position[:, i, :],
                    anim_poses.quaternion[:, i, :]).multiply(op)
        anim_poses.position[:, i, :]   = np_.position
        anim_poses.quaternion[:, i, :] = np_.quaternion

    # EE path
    ee_all, _ = _get_ee_pose(kin_model, all_q, ee_link)
    ee_np = ee_all.cpu().numpy()
    base_np = all_q[:, :2].cpu().numpy()

    # ── markers ───────────────────────────────────────────────────────
    cuboids = []
    spheres = []

    # visual ground at z=0
    cuboids.append(Cuboid(name="vis_ground",
                          pose=[0, 0, -0.005, 1, 0, 0, 0],
                          dims=[5.0, 5.0, 0.01],
                          color=[0.85, 0.85, 0.85, 0.2]))

    # home EE
    spheres.append(Sphere(name="home_ee",
                          pose=list(home_ee_np) + [1, 0, 0, 0],
                          radius=0.04, color=COL_HOME))

    off = 0
    for ti, (traj, goal, lab) in enumerate(zip(trajs, goals, labels)):
        n = traj.position.shape[0]
        c  = COL_WAYPOINTS[ti % len(COL_WAYPOINTS)]
        pc = COL_PATH[ti % len(COL_PATH)]
        safe = lab.replace("+", "p").replace("-", "m") \
                  .replace(" ", "_").replace("(", "").replace(")", "")

        # goal sphere
        gp = goal.position[0].cpu().numpy()
        spheres.append(Sphere(name=f"goal_{ti}_{safe}",
                              pose=list(map(float, gp)) + [1, 0, 0, 0],
                              radius=0.04, color=c))

        # dense EE trail
        step = max(1, n // 40)
        for idx in range(0, n, step):
            ai = off + idx
            spheres.append(Sphere(
                name=f"ee_{ti}_{idx:04d}",
                pose=[float(ee_np[ai, 0]), float(ee_np[ai, 1]),
                      float(ee_np[ai, 2]), 1, 0, 0, 0],
                radius=0.010, color=pc))
        # final
        spheres.append(Sphere(
            name=f"ee_{ti}_end",
            pose=[float(ee_np[off+n-1, 0]), float(ee_np[off+n-1, 1]),
                  float(ee_np[off+n-1, 2]), 1, 0, 0, 0],
            radius=0.010, color=pc))

        # base trail on ground
        for idx in range(0, n, max(1, n // 20)):
            ai = off + idx
            spheres.append(Sphere(
                name=f"base_{ti}_{idx:04d}",
                pose=[float(base_np[ai, 0]), float(base_np[ai, 1]),
                      0.01, 1, 0, 0, 0],
                radius=0.015, color=[1.0, 0.85, 0.0, 0.5]))
        off += n

    # world obstacles → cuboid list
    if world_cfg and world_cfg.cuboid:
        cuboids.extend(world_cfg.cuboid)

    mw = WorldConfig(cuboid=cuboids, sphere=spheres)

    # ── write stage ───────────────────────────────────────────────────
    if os.path.exists(save_path):
        os.remove(save_path)
    uh = UsdHelper()
    uh.create_stage(save_path, timesteps=all_q.shape[0],
                    dt=dt, interpolation_steps=1, base_frame=base_frame)
    uh.add_world_to_stage(mw, base_frame=base_frame)
    uh.create_animation(robot_world, anim_poses, base_frame,
                        robot_frame="robot")
    uh.write_stage_to_file(save_path, flatten=False)


# ═══════════════════════════════════════════════════════════════════════
#  Spray waypoint generation
# ═══════════════════════════════════════════════════════════════════════
def _generate_spray_waypoints(home_pos, wall_y, spray_quat):
    """Generate Z-letter waypoints on the wall surface.

    The Z shape has 4 waypoints:
        1 (top-left) ──→ 2 (top-right)     top stroke
        2 (top-right) ──↙ 3 (bottom-left)  diagonal
        3 (bottom-left) ──→ 4 (bottom-right) bottom stroke

    Returns list of (label, pos_np, quat_np) tuples.
    The EE is offset from the wall by SPRAY_STANDOFF in -Y direction
    (sprayer faces +Y into the wall).
    """
    ee_y = wall_y - WALL_THICKNESS / 2.0 - SPRAY_STANDOFF
    center_x = home_pos[0]
    top_z = home_pos[2] + Z_HEIGHT / 2.0
    bot_z = home_pos[2] - Z_HEIGHT / 2.0
    left_x = center_x - SWEEP_HALF_WIDTH
    right_x = center_x + SWEEP_HALF_WIDTH

    waypoints = [
        ("Z_top_left",     np.array([left_x,  ee_y, top_z]), spray_quat),
        ("Z_top_right",    np.array([right_x, ee_y, top_z]), spray_quat),
        ("Z_bottom_left",  np.array([left_x,  ee_y, bot_z]), spray_quat),
        ("Z_bottom_right", np.array([right_x, ee_y, bot_z]), spray_quat),
    ]
    return waypoints


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    interpolation_dt = 0.02

    # load robot config
    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg["kinematics"]["collision_spheres"] = _SPHERES_PATH

    # Slow down motion by 0.25x for clearer visualization
    cspace = robot_cfg["kinematics"]["cspace"]
    cspace["max_acceleration"] = cspace.get("max_acceleration", 15.0) * 0.25
    cspace["max_jerk"] = cspace.get("max_jerk", 500.0) * 0.25

    # ── Step 1: FK at home to find EE pose ────────────────────────────
    kin, kcfg = _build_kin_model(robot_cfg, tensor_args)
    ee_link = kcfg["kinematics"]["ee_link"]
    home_q = tensor_args.to_device([HOME_Q])
    home_pos, home_quat = _get_ee_pose(kin, home_q, ee_link)
    hp = home_pos[0].cpu().numpy()
    hq = home_quat[0].cpu().numpy()

    print(f"\n=== Home EE (eoa_tool_change) ===")
    print(f"  pos : ({hp[0]:.4f}, {hp[1]:.4f}, {hp[2]:.4f})")
    print(f"  quat: ({hq[0]:.4f}, {hq[1]:.4f}, {hq[2]:.4f}, {hq[3]:.4f})")

    # ── Step 2: Place the wall in front of the EE ─────────────────────
    # Wall is placed ~0.4 m in front of the EE along +Y
    wall_center_y = hp[1] + 0.40
    wall_center_z = hp[2]
    world_cfg = _build_world(wall_center_y, wall_center_z)
    print(f"\n=== Wall placement ===")
    print(f"  center: (0.0, {wall_center_y:.3f}, {wall_center_z:.3f})")
    print(f"  size  : {WALL_WIDTH} x {WALL_THICKNESS} x {WALL_HEIGHT}")
    print(f"  Obstacles: {[c.name for c in world_cfg.cuboid]}")

    # ── Step 3: Build MotionGen ───────────────────────────────────────
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg, world_cfg, tensor_args,
        trajopt_tsteps=34,
        interpolation_steps=5000,
        num_ik_seeds=100,
        num_trajopt_seeds=12,
        grad_trajopt_iters=500,
        trajopt_dt=0.5,
        interpolation_dt=interpolation_dt,
        evaluate_interpolated_trajectory=True,
        js_trajopt_dt=0.5,
        js_trajopt_tsteps=34,
        self_collision_check=True,
        self_collision_opt=True,
        collision_activation_distance=0.025,
    )
    mg = MotionGen(mg_cfg)
    print("Warming up ...")
    mg.warmup()

    # ── Step 4: Generate spray waypoints ──────────────────────────────
    spray_quat = hq.copy()

    waypoints = _generate_spray_waypoints(hp, wall_center_y, spray_quat)
    print(f"\n=== Spray waypoints ({len(waypoints)} total) ===")
    for lab, pos, _ in waypoints:
        print(f"  {lab:25s}  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print(f"\n  Self-collision: ON  |  World collision: ON")
    print(f"  Spray standoff: {SPRAY_STANDOFF} m from wall surface\n")

    # ── Step 5: Sequential planning ───────────────────────────────────
    trajs  = []
    goals  = []
    labels = []
    current_q = home_q.clone()
    n_legs = len(waypoints) + 1  # +1 for return-to-home

    for wi, (lab, pos, quat) in enumerate(waypoints):
        leg_num = wi + 1
        print(f"[{leg_num}/{n_legs}] {'Home' if wi == 0 else waypoints[wi-1][0]} -> {lab} ...")
        goal = _make_pose(pos, quat, tensor_args)
        result = mg.plan_single(
            JointState.from_position(current_q, joint_names=JOINT_NAMES), goal)
        success = result.success.item()
        print(f"  success={success}, time={result.total_time:.3f}s")
        if success:
            traj = result.get_interpolated_plan()
            trajs.append(traj)
            goals.append(goal)
            labels.append(f"leg{leg_num}_{lab}")
            print(f"  steps={traj.position.shape[0]}")
            plot_traj(traj, interpolation_dt, JOINT_NAMES,
                      f"Leg {leg_num} -> {lab}",
                      f"robotv1_1_spray_leg{leg_num}.png")
            current_q = traj.position[-1:].clone()
        else:
            print(f"  FAILED — skipping leg {leg_num}")

    # ── Return to home ────────────────────────────────────────────────
    if trajs:
        print(f"[{n_legs}/{n_legs}] {waypoints[-1][0]} -> Home ...")
        home_goal = _make_pose(hp, hq, tensor_args)
        result = mg.plan_single(
            JointState.from_position(current_q, joint_names=JOINT_NAMES),
            home_goal)
        success = result.success.item()
        print(f"  success={success}, time={result.total_time:.3f}s")
        if success:
            traj = result.get_interpolated_plan()
            trajs.append(traj)
            goals.append(home_goal)
            labels.append("return_home")
            print(f"  steps={traj.position.shape[0]}")
            plot_traj(traj, interpolation_dt, JOINT_NAMES,
                      f"Leg {n_legs} -> Home",
                      f"robotv1_1_spray_leg{n_legs}.png")
        else:
            print(f"  FAILED — skipping return to home")

    if not trajs:
        print("\nNo successful legs — aborting.")
        return

    # ── Combined plot ─────────────────────────────────────────────────
    comb = JointState(
        position=torch.cat([t.position for t in trajs]),
        velocity=torch.cat([t.velocity for t in trajs]),
        acceleration=torch.cat([t.acceleration for t in trajs]),
        jerk=torch.cat([t.jerk for t in trajs]),
    )
    plot_traj(comb, interpolation_dt, JOINT_NAMES,
              "Combined — Spray Z Pattern", "robotv1_1_traj_combined.png")

    print(f"\n=== Summary ===")
    print(f"  Successful legs: {len(trajs)} / {n_legs}")
    print(f"  Total frames   : {comb.position.shape[0]}")

    # ── USD ───────────────────────────────────────────────────────────
    if USD_AVAILABLE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        usd_path = f"robotv1_1_demo_{ts}.usd"
        _save_usd(robot_cfg, trajs, goals, labels, hp,
                  world_cfg, interpolation_dt, usd_path,
                  "/world_base", tensor_args)
        full = os.path.join(_SCRIPT_DIR, usd_path)
        print(f"\nUSD saved: {full}")
        print(f"  Frames : {comb.position.shape[0]}")
        print(f"  Legs   : {len(trajs)}")
    else:
        print("\nSkipping USD (usd-core not installed)")


if __name__ == "__main__":
    main()
