"""RobotV1.1 — writes "ORIGIN" on a large wall via Cartesian motion.

The robot starts at the eoa_tool_change home pose, does FK to find the EE pose,
places a large wall in front of the sprayer, then plans Cartesian motions to
write the word "ORIGIN" letter by letter.

Each letter is a sequence of pen-down strokes (EE near wall) with pen-up
transitions (EE pulled back) between strokes / letters. Waypoints are revealed
live during execution — not shown beforehand.

Self-collision and world-collision are both enabled.
The USD contains the robot animation, the wall, ground, waypoint markers,
and dense EE/base path traces (green = pen-down writing).

Usage:
    cd ~/IsaacLab/scripts/standalone/curobo_v1_1
    python robotv1_1_write_origin.py
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

# ── wall & letter parameters ──────────────────────────────────────────
SPRAY_STANDOFF = 0.15          # pen-down: distance from wall surface to EE
PEN_UP_EXTRA = 0.12            # pen-up: additional pullback from wall
WALL_THICKNESS = 0.04          # thin wall
WALL_WIDTH = 2.0               # wall X extent
WALL_HEIGHT = 1.5              # wall Z extent
LETTER_HEIGHT = 0.28           # height of each letter
LETTER_WIDTH = 0.18            # width of each letter
LETTER_SPACING = 0.06          # gap between letters
LINE_INTERP_STEP = 0.03       # max Cartesian distance between pen-down waypoints (m)

# ── letter stroke definitions ─────────────────────────────────────────
# Each letter = list of strokes. Each stroke = list of (dx, dz) in [0,1].
# (0,0) = bottom-left of letter cell, (1,1) = top-right.
# Pen is DOWN within a stroke, UP between strokes and between letters.
LETTERS = {
    "O": [[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]],
    "R": [
        [(0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0.5)],  # vertical + bump
        [(0, 0.5), (1, 0)],                               # kick leg
    ],
    "I": [[(0.5, 0), (0.5, 1)]],
    "G": [[(1, 1), (0, 1), (0, 0), (1, 0), (1, 0.5)]],
    "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
}

WORD = "ORIGIN"

# ── colours ────────────────────────────────────────────────────────────
COL_HOME      = [1.0, 1.0, 1.0, 1.0]
COL_WALL      = [0.55, 0.55, 0.60, 0.7]
COL_GROUND    = [0.80, 0.80, 0.80, 0.25]
COL_PEN_DOWN  = [0.0, 0.85, 0.0, 0.85]   # green = writing
COL_PEN_UP    = [1.0, 0.4, 0.0, 0.4]      # faint orange = transit
COL_WAYPOINTS = [
    [1.0, 0.3, 0.0, 1.0],
    [0.0, 0.85, 0.0, 1.0],
    [0.0, 0.35, 1.0, 1.0],
    [0.85, 0.0, 0.85, 1.0],
    [1.0, 0.85, 0.0, 1.0],
    [0.0, 0.85, 0.85, 1.0],
    [1.0, 0.0, 0.0, 1.0],
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
    print(f"    Plot saved: {file_name}")


# ═══════════════════════════════════════════════════════════════════════
#  USD export
# ═══════════════════════════════════════════════════════════════════════
INK_COLOR = [1.0, 0.15, 0.0, 1.0]   # bright red ink
INK_RADIUS = 0.002
INK_DENSITY = 30  # spheres per line segment


def _save_usd(robot_cfg, trajs, goals, labels, is_pen_down, home_ee_np,
              world_cfg, wall_surface_y, dt, save_path, base_frame,
              tensor_args):
    from pxr import UsdGeom

    kin_model, usd_cfg = _build_kin_model(robot_cfg, tensor_args)
    ee_link = usd_cfg["kinematics"]["ee_link"]

    all_q = torch.cat([t.position for t in trajs], dim=0)
    total_frames = all_q.shape[0]

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

    # ── static markers (wall, ground, home) ───────────────────────────
    cuboids = []
    spheres = []

    cuboids.append(Cuboid(name="vis_ground",
                          pose=[0, 0, -0.005, 1, 0, 0, 0],
                          dims=[5.0, 5.0, 0.01],
                          color=[0.85, 0.85, 0.85, 0.2]))

    spheres.append(Sphere(name="home_ee",
                          pose=list(home_ee_np) + [1, 0, 0, 0],
                          radius=0.04, color=COL_HOME))

    # ── EE path trace + goal waypoint markers ────────────────────────
    ee_all, _ = _get_ee_pose(kin_model, all_q, ee_link)
    ee_np = ee_all.cpu().numpy()
    base_np = all_q[:, :2].cpu().numpy()

    off = 0
    for ti, (traj, goal, lab, pen) in enumerate(
            zip(trajs, goals, labels, is_pen_down)):
        n = traj.position.shape[0]
        c = COL_PEN_DOWN if pen else COL_PEN_UP
        safe = lab.replace("+", "p").replace("-", "m") \
                  .replace(" ", "_").replace("(", "").replace(")", "")

        # goal sphere
        gp = goal.position[0].cpu().numpy()
        spheres.append(Sphere(name=f"goal_{ti}_{safe}",
                              pose=list(map(float, gp)) + [1, 0, 0, 0],
                              radius=0.0004, color=c))

        # dense EE trail (pen-down = green, pen-up = faint orange)
        trail_col = COL_PEN_DOWN if pen else COL_PEN_UP
        step = max(1, n // 40)
        for idx in range(0, n, step):
            ai = off + idx
            spheres.append(Sphere(
                name=f"ee_{ti}_{idx:04d}",
                pose=[float(ee_np[ai, 0]), float(ee_np[ai, 1]),
                      float(ee_np[ai, 2]), 1, 0, 0, 0],
                radius=0.0004, color=trail_col))
        # final point
        spheres.append(Sphere(
            name=f"ee_{ti}_end",
            pose=[float(ee_np[off+n-1, 0]), float(ee_np[off+n-1, 1]),
                  float(ee_np[off+n-1, 2]), 1, 0, 0, 0],
            radius=0.006, color=trail_col))

        # base trail on ground
        for idx in range(0, n, max(1, n // 20)):
            ai = off + idx
            spheres.append(Sphere(
                name=f"base_{ti}_{idx:04d}",
                pose=[float(base_np[ai, 0]), float(base_np[ai, 1]),
                      0.01, 1, 0, 0, 0],
                radius=0.0004, color=[1.0, 0.85, 0.0, 0.5]))
        off += n

    if world_cfg and world_cfg.cuboid:
        cuboids.extend(world_cfg.cuboid)

    static_world = WorldConfig(cuboid=cuboids, sphere=spheres)

    # ── compute cumulative frame offsets per leg ─────────────────────
    frame_end = []    # frame index where each leg ends
    off = 0
    for traj in trajs:
        off += traj.position.shape[0]
        frame_end.append(off)

    # ── build ink segments info (for animated spheres) ───────────────
    # Each ink segment connects two consecutive pen-down goals on the wall
    ink_segments = []  # list of (start_pt, end_pt, reveal_frame)
    prev_pen_wall = None
    for ti, (goal, pen) in enumerate(zip(goals, is_pen_down)):
        if pen:
            gp = goal.position[0].cpu().numpy()
            wall_pt = np.array([float(gp[0]), wall_surface_y, float(gp[2])])
            if prev_pen_wall is not None:
                ink_segments.append((prev_pen_wall, wall_pt, frame_end[ti]))
            prev_pen_wall = wall_pt
        else:
            prev_pen_wall = None

    # ── create ink spheres (added to a WorldConfig for placement) ────
    ink_spheres = []
    ink_names = []          # prim names for visibility animation
    ink_reveal_frames = []  # frame at which each sphere becomes visible
    ink_id = 0
    for seg_start, seg_end, reveal_frame in ink_segments:
        for si in range(INK_DENSITY + 1):
            alpha = si / INK_DENSITY
            pt = seg_start + alpha * (seg_end - seg_start)
            name = f"ink_{ink_id:04d}"
            ink_spheres.append(Sphere(
                name=name,
                pose=[float(pt[0]), float(pt[1]), float(pt[2]),
                      1, 0, 0, 0],
                radius=INK_RADIUS, color=INK_COLOR))
            ink_names.append(name)
            ink_reveal_frames.append(reveal_frame)
            ink_id += 1

    ink_world = WorldConfig(sphere=ink_spheres) if ink_spheres else None

    # ── write stage ───────────────────────────────────────────────────
    if os.path.exists(save_path):
        os.remove(save_path)
    uh = UsdHelper()
    uh.create_stage(save_path, timesteps=total_frames,
                    dt=dt, interpolation_steps=1, base_frame=base_frame)
    uh.add_world_to_stage(static_world, base_frame=base_frame)
    if ink_world:
        uh.add_world_to_stage(ink_world, base_frame=base_frame)
    uh.create_animation(robot_world, anim_poses, base_frame,
                        robot_frame="robot")

    # ── animate ink visibility — hidden until robot paints each segment
    stage = uh.stage
    for name, reveal in zip(ink_names, ink_reveal_frames):
        prim_path = f"{base_frame}/{name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(prim)
        vis_attr = imageable.GetVisibilityAttr()
        vis_attr.Set(UsdGeom.Tokens.invisible, 0)
        vis_attr.Set(UsdGeom.Tokens.inherited, reveal)

    uh.write_stage_to_file(save_path, flatten=False)
    print(f"    Ink spheres: {ink_id}  ({len(ink_segments)} segments, "
          f"animated visibility)")


# ═══════════════════════════════════════════════════════════════════════
#  Letter waypoint generation
# ═══════════════════════════════════════════════════════════════════════
def _generate_word_waypoints(word, center_x, ee_y_down, ee_y_up, center_z,
                             spray_quat):
    """Generate waypoints to write a word on the wall.

    Returns list of (label, pos_np, quat_np, is_pen_down) tuples.
    Pen-down waypoints are at ee_y_down, pen-up at ee_y_up.
    """
    n_letters = len(word)
    total_width = n_letters * LETTER_WIDTH + (n_letters - 1) * LETTER_SPACING
    start_x = center_x - total_width / 2.0
    bot_z = center_z - LETTER_HEIGHT / 2.0

    waypoints = []
    for li, ch in enumerate(word):
        letter_x0 = start_x + li * (LETTER_WIDTH + LETTER_SPACING)
        strokes = LETTERS[ch]

        for si, stroke in enumerate(strokes):
            # pen-up: move to first point of this stroke (pulled back)
            sx, sz = stroke[0]
            up_pos = np.array([letter_x0 + sx * LETTER_WIDTH,
                               ee_y_up,
                               bot_z + sz * LETTER_HEIGHT])
            waypoints.append(
                (f"{ch}_{li}_s{si}_penup", up_pos, spray_quat, False))

            # pen-down: trace each segment with dense Cartesian waypoints
            # so IK + linear joint interp gives straight-line EE motion
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

    # Increase self-collision buffers (YAML defaults are all 0.0)
    kin_cfg = robot_cfg["kinematics"]
    scb = kin_cfg.get("self_collision_buffer", {})
    for link in scb:
        scb[link] = 0.04
    kin_cfg["self_collision_buffer"] = scb
    kin_cfg["collision_sphere_buffer"] = 0.01

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
    wall_center_y = hp[1] + 0.40
    wall_center_z = hp[2]
    world_cfg = _build_world(wall_center_y, wall_center_z)
    print(f"\n=== Wall placement ===")
    print(f"  center: (0.0, {wall_center_y:.3f}, {wall_center_z:.3f})")
    print(f"  size  : {WALL_WIDTH} x {WALL_THICKNESS} x {WALL_HEIGHT}")

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
        collision_activation_distance=0.10,
    )
    mg = MotionGen(mg_cfg)
    print("Warming up ...")
    mg.warmup()

    # ── Step 4: Generate letter waypoints ─────────────────────────────
    spray_quat = hq.copy()
    wall_surface_y = wall_center_y - WALL_THICKNESS / 2.0
    ee_y_down = wall_surface_y - SPRAY_STANDOFF
    ee_y_up = wall_surface_y - SPRAY_STANDOFF - PEN_UP_EXTRA

    waypoints = _generate_word_waypoints(
        WORD, hp[0], ee_y_down, ee_y_up, hp[2], spray_quat)

    n_legs = len(waypoints) + 1  # +1 for return-to-home
    # Verify self-collision is active
    sc_active = (mg.trajopt_solver is not None and
                 hasattr(mg.trajopt_solver, 'rollout_fn') and
                 mg.trajopt_solver.rollout_fn.constraint_fn is not None)
    print(f"\n=== Writing \"{WORD}\" on the wall ===")
    print(f"  Total legs: {n_legs}  (waypoints revealed during execution)")
    print(f"  Self-collision: {'ACTIVE' if sc_active else 'OFF'}  |  "
          f"World collision: ON")
    print(f"  Self-collision buffer: 0.04 per link  |  "
          f"Sphere buffer: 0.01  |  Activation: 0.10 m")
    print(f"  Spray standoff: {SPRAY_STANDOFF} m  |  Pen-up extra: {PEN_UP_EXTRA} m\n")

    # ── Step 5: Sequential planning ──────────────────────────────────
    #   pen-up  → plan_single (trajectory optimizer, free path)
    #   pen-down → IK at each dense waypoint + linear joint interpolation
    #              → guarantees straight-line Cartesian EE motion (spraying)
    #              Waypoints are 30 mm apart, standoff is 150 mm from wall,
    #              so linear joint interp between close IK solutions is safe.
    trajs       = []
    goals       = []
    labels      = []
    pen_flags   = []
    current_q   = home_q.clone()
    fail_count  = 0

    # Steps for linear joint interpolation between consecutive IK solutions
    JOINT_INTERP_STEPS = 50

    for wi, (lab, pos, quat, pen) in enumerate(waypoints):
        leg_num = wi + 1
        pen_str = "PEN-DOWN" if pen else "pen-up"
        print(f"[{leg_num}/{n_legs}] {pen_str:>8s}  -> {lab:20s}  "
              f"({pos[0]:+.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        goal = _make_pose(pos, quat, tensor_args)

        if not pen:
            # ── Pen-up: plan_single (path shape doesn't matter) ──────
            result = mg.plan_single(
                JointState.from_position(current_q, joint_names=JOINT_NAMES),
                goal)
            if result.success.item():
                traj = result.get_interpolated_plan()
                trajs.append(traj)
                goals.append(goal)
                labels.append(f"leg{leg_num}_{lab}")
                pen_flags.append(False)
                print(f"           OK  steps={traj.position.shape[0]}  "
                      f"time={result.total_time:.3f}s")
                current_q = traj.position[-1:].clone()
            else:
                fail_count += 1
                print(f"           FAILED  (skipped)")
        else:
            # ── Pen-down: IK + linear joint interp = straight line ───
            seed = current_q.view(1, 1, -1)
            ik_result = mg.solve_ik(goal, seed_config=seed, return_seeds=1)

            if not ik_result.success.any():
                fail_count += 1
                print(f"           IK FAILED  (skipped)")
                continue

            goal_q = ik_result.solution[ik_result.success][0:1]

            # Linear interpolation in joint space between two close
            # configs → near-perfect straight line in Cartesian space
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

    # ── Return to home ────────────────────────────────────────────────
    if trajs:
        print(f"[{n_legs}/{n_legs}]   pen-up  -> Home")
        home_goal = _make_pose(hp, hq, tensor_args)
        result = mg.plan_single(
            JointState.from_position(current_q, joint_names=JOINT_NAMES),
            home_goal)
        success = result.success.item()
        if success:
            traj = result.get_interpolated_plan()
            trajs.append(traj)
            goals.append(home_goal)
            labels.append("return_home")
            pen_flags.append(False)
            print(f"           OK  steps={traj.position.shape[0]}  "
                  f"time={result.total_time:.3f}s")
        else:
            fail_count += 1
            print(f"           FAILED  (skipped)")

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
              f"Combined — \"{WORD}\" on wall",
              "robotv1_1_origin_traj_combined.png")

    print(f"\n=== Summary ===")
    print(f"  Word         : {WORD}")
    print(f"  Successful   : {len(trajs)} / {n_legs}")
    print(f"  Failed       : {fail_count}")
    print(f"  Total frames : {comb.position.shape[0]}")

    # ── USD ───────────────────────────────────────────────────────────
    if USD_AVAILABLE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        usd_path = f"robotv1_1_origin_{ts}.usd"
        _save_usd(robot_cfg, trajs, goals, labels, pen_flags, hp,
                  world_cfg, wall_surface_y, interpolation_dt, usd_path,
                  "/world_base", tensor_args)
        full = os.path.join(_SCRIPT_DIR, usd_path)
        print(f"\nUSD saved: {full}")
        print(f"  Frames : {comb.position.shape[0]}")
        print(f"  Legs   : {len(trajs)}")
    else:
        print("\nSkipping USD (usd-core not installed)")


if __name__ == "__main__":
    main()
