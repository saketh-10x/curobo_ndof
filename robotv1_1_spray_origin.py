"""RobotV1.1 — cuRobo letter-tracing + Warp spray to paint "ORIGIN" on a wall.

Phase 1: cuRobo plans "ORIGIN" letter waypoints (pen-up: trajectory optimizer,
         pen-down: IK + linear joint interp for straight Cartesian strokes).
Phase 2: Extract EE path + build per-frame pen_down mask.
Phase 3: Warp spray simulation (emit particles only during pen-down frames).
Phase 4: pxr adds the animated robot mesh to the spray USD.

Usage (standalone — no Isaac Sim required):
    cd ~/IsaacLab
    python scripts/standalone/curobo_v1_1/robotv1_1_spray_origin.py
"""

import math
import os
import sys
import time

import numpy as np
import torch

# ── Warp spray utils ────────────────────────────────────────────────
_SPRAY_ASSIGNMENT_DIR = "/home/sim-system/Downloads/spray_simulation_assignment"
sys.path.insert(0, _SPRAY_ASSIGNMENT_DIR)

import warp as wp
from utils.particles import ParticleSystem
from utils.mesh import WallMesh
from utils.renderer import Renderer

# ── cuRobo imports ──────────────────────────────────────────────────
from curobo.types.state import JointState
from curobo.types.math import Pose as CuroboPose
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.geom.types import Cuboid, WorldConfig
from curobo.util_file import load_yaml

# ── Paths ───────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")
_SPHERES_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_spheres.yml")
_ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "usd_generated", "robotv1_1_sprayer.usd")
_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "spray_output")

# ── Joint configuration (10-DOF) ───────────────────────────────────
JOINT_NAMES = [
    "base_x", "base_y", "base_theta", "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
HOME_ARM = [0.0, 0.0, -1.5708, -3.1416, -1.5708, 3.1416]
HOME_Q = [0.0, 0.0, 0.0, 0.0] + HOME_ARM

# ── Wall & letter parameters ─────────────────────────────────────────
SPRAY_STANDOFF = 0.15          # pen-down: distance from wall surface to EE
PEN_UP_EXTRA = 0.12            # pen-up: additional pullback from wall
WALL_THICKNESS = 0.04
WALL_WIDTH = 2.0
WALL_HEIGHT = 1.5
LETTER_HEIGHT = 0.28           # height of each letter
LETTER_WIDTH = 0.18            # width of each letter
LETTER_SPACING = 0.06          # gap between letters
LINE_INTERP_STEP = 0.03        # max Cartesian distance between pen-down waypoints (m)

# ── Letter stroke definitions ────────────────────────────────────────
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

# Steps for linear joint interpolation between consecutive IK solutions
JOINT_INTERP_STEPS = 50

# ── Visual link names to animate in USD ─────────────────────────────
VISUAL_LINKS = [
    "base_link", "base_link_inertia",
    "liftkit_mid", "liftkit_top",
    "shoulder_link", "upper_arm_link", "forearm_link",
    "forearm_clip_1", "forearm_clip_2", "upperarm_clip",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
    "tool_mount", "flange", "flange_mount",
    "steering", "steering_2", "steering_3",
    "wheel", "wheel_2", "wheel_3",
    "tool0",
]


# ═══════════════════════════════════════════════════════════════════
#  cuRobo Helpers
# ═══════════════════════════════════════════════════════════════════
def compute_fk(motion_gen, q):
    if q.dim() == 1:
        q = q.unsqueeze(0)
    fk = motion_gen.kinematics.get_state(q)
    return fk.ee_position[0].clone(), fk.ee_quaternion[0].clone()


def compute_fk_batch_ee(motion_gen, q_batch):
    if q_batch.dim() == 1:
        q_batch = q_batch.unsqueeze(0)
    fk = motion_gen.kinematics.get_state(q_batch)
    return fk.ee_position.clone()


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


def build_motion_gen(wall_center_y, wall_center_z):
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"),
                                   dtype=torch.float32)
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

    world_cfg = WorldConfig(cuboid=[
        Cuboid(name="ground",
               pose=[0, 0, -0.50, 1, 0, 0, 0],
               dims=[6.0, 6.0, 0.02]),
        Cuboid(name="spray_wall",
               pose=[0.0, wall_center_y, wall_center_z, 1, 0, 0, 0],
               dims=[WALL_WIDTH, WALL_THICKNESS, WALL_HEIGHT]),
    ])

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
    print("[INFO] Warming up cuRobo MotionGen ...")
    mg.warmup()
    print("[INFO] cuRobo warmup complete.")
    return mg, tensor_args


# ═══════════════════════════════════════════════════════════════════
#  Letter Waypoint Generation
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
#  Coordinate Mapping: cuRobo -> Warp
# ═══════════════════════════════════════════════════════════════════
def curobo_to_warp(positions_np, wall_surface_y, wall_center_x, wall_center_z):
    """Map cuRobo positions to Warp coordinates.

    cuRobo: X=horizontal, Y=depth(toward wall), Z=vertical.
    Warp:   X=horizontal, Y=vertical, Z=depth(away from wall, wall at Z=0).
    """
    warp_pos = np.zeros_like(positions_np)
    warp_pos[..., 0] = positions_np[..., 0] - wall_center_x
    warp_pos[..., 1] = positions_np[..., 2] - wall_center_z
    warp_pos[..., 2] = wall_surface_y - positions_np[..., 1]
    return warp_pos


# ═══════════════════════════════════════════════════════════════════
#  yourdfpy FK for All Links
# ═══════════════════════════════════════════════════════════════════
def compute_all_link_fk(joint_trajectories, joint_names, urdf_path, link_names):
    """Use yourdfpy to compute FK for all visual links at each frame.

    Args:
        joint_trajectories: (N, 10) numpy array of joint positions.
        joint_names: list of 10 joint names (matching trajectory columns).
        urdf_path: path to URDF file.
        link_names: list of link names to compute FK for.

    Returns:
        dict mapping link_name -> (N, 4, 4) array of world transforms.
    """
    import yourdfpy

    robot = yourdfpy.URDF.load(urdf_path)
    n_frames = len(joint_trajectories)

    link_transforms = {name: np.zeros((n_frames, 4, 4)) for name in link_names}

    for fi in range(n_frames):
        # Build configuration dict
        cfg = {}
        for ji, jn in enumerate(joint_names):
            cfg[jn] = float(joint_trajectories[fi, ji])
        # liftkit_top mirrors liftkit_mid
        cfg["liftkit_top"] = cfg.get("liftkit_mid", 0.0)

        robot.update_cfg(cfg)

        for ln in link_names:
            try:
                tf = robot.get_transform(ln)
                link_transforms[ln][fi] = tf
            except Exception:
                link_transforms[ln][fi] = np.eye(4)

        if (fi + 1) % 500 == 0 or fi == n_frames - 1:
            print(f"    FK frame {fi + 1}/{n_frames}")

    return link_transforms


# ═══════════════════════════════════════════════════════════════════
#  Add Robot Mesh to Spray USD
# ═══════════════════════════════════════════════════════════════════
def add_robot_to_usd(spray_usd_path, robot_usd_path,
                     link_transforms_warp, save_frame_indices, frame_rate):
    """Add the animated robot mesh to the spray USD using pxr.

    Uses resetXformStack so each link gets world-space transforms
    independent of the USD hierarchy.
    """
    from pxr import Usd, UsdGeom, Gf, Sdf

    stage = Usd.Stage.Open(spray_usd_path)

    # Add robot as a reference under /Robot
    robot_root = stage.OverridePrim("/Robot")
    robot_root.GetReferences().AddReference(
        Sdf.Reference(robot_usd_path))

    animated_count = 0

    for link_name, transforms in link_transforms_warp.items():
        prim_path = f"/Robot/{link_name}"
        prim = stage.OverridePrim(prim_path)
        if not prim or not prim.IsValid():
            continue

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue

        # Reset xform stack: this link ignores parent transforms
        xformable.SetResetXformStack(True)
        xformable.ClearXformOpOrder()

        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp(
            precision=UsdGeom.XformOp.PrecisionDouble)

        for fi, frame_idx in enumerate(save_frame_indices):
            if fi >= len(transforms):
                break
            t = frame_idx / frame_rate

            tf = transforms[fi]  # 4x4 homogeneous transform
            pos = tf[:3, 3]
            rot_mat = tf[:3, :3]

            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), t)

            # Convert 3x3 rotation matrix to quaternion
            quat = _rotation_matrix_to_quat(rot_mat)
            orient_op.Set(Gf.Quatd(float(quat[0]), float(quat[1]),
                                    float(quat[2]), float(quat[3])), t)

        animated_count += 1

    stage.Save()
    print(f"  Robot mesh added to USD: {animated_count} links animated")


def _rotation_matrix_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _transform_4x4_curobo_to_warp(tf_curobo, wall_surface_y, cx, cz):
    """Transform a 4x4 homogeneous matrix from cuRobo to Warp coordinates.

    cuRobo axes: X=right, Y=forward(toward wall), Z=up
    Warp axes:   X=right, Y=up, Z=back(away from wall)
    """
    R_frame = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ], dtype=np.float64)

    tf_warp = np.eye(4)
    tf_warp[:3, :3] = R_frame @ tf_curobo[:3, :3]
    p = tf_curobo[:3, 3]
    tf_warp[0, 3] = p[0] - cx
    tf_warp[1, 3] = p[2] - cz
    tf_warp[2, 3] = wall_surface_y - p[1]
    return tf_warp


# ═══════════════════════════════════════════════════════════════════
#  Warp Spray Simulation (pen-down-only emission)
# ═══════════════════════════════════════════════════════════════════
def run_spray_simulation(nozzle_positions, pen_down_mask, output_dir):
    """Run the Warp GPU spray simulation along the nozzle path.

    Particles are only emitted during pen-down frames.
    """
    wp.init()
    os.makedirs(output_dir, exist_ok=True)

    wall_config = {
        "width": 2.5,
        "height": 2.0,
        "resolution": 200,
        "base_color": [0.92, 0.92, 0.92],
        "paint_color": [0.85, 0.12, 0.12],
        "particle_radius": 0.035,
    }
    wall = WallMesh(wall_config)
    print(f"[WARP] Wall: {wall_config['width']}x{wall_config['height']}m, "
          f"res={wall_config['resolution']}")

    sim_config = {"max_particles": 8000, "time_step": 1.0 / 60.0}
    physics_config = {
        "gravity": [0.0, 0.0, -0.2],
        "air_resistance": 0.97,
        "particle_lifetime": 2.0,
        "velocity_variation": [0.90, 1.10],
        "initial_velocity": 15.0,
    }
    particles = ParticleSystem(sim_config, physics_config)

    render_config = {
        "directory": output_dir,
        "save_frequency": 5,
        "frame_rate": 60,
    }
    renderer = Renderer(render_config)

    particles_per_frame = 40
    cone_angle = 0.15
    spray_dir = np.array([0.0, 0.0, -1.0])
    n_frames = len(nozzle_positions)
    save_every = max(1, n_frames // 200)
    save_frame_indices = []

    pen_down_frames = int(pen_down_mask.sum())
    print(f"[WARP] Simulating {n_frames} frames "
          f"({pen_down_frames} pen-down, {n_frames - pen_down_frames} pen-up) ...")
    t0 = time.time()

    for frame in range(n_frames):
        nozzle_pos = nozzle_positions[frame]

        # Only emit particles during pen-down (letter stroke) frames
        if pen_down_mask[frame]:
            particles.emit(
                nozzle_pos=nozzle_pos,
                spray_dir=spray_dir,
                num_particles=particles_per_frame,
                cone_angle=cone_angle,
            )

        # Always update physics and paint
        particles.update()
        wall.update_paint(
            particles.positions, particles.velocities, particles.active,
        )

        if frame % save_every == 0 or frame == n_frames - 1:
            save_frame_indices.append(frame)

            active_np = particles.active.numpy()
            pos_np = particles.positions.numpy()
            active_mask = active_np > 0
            active_positions = pos_np[active_mask] if active_mask.any() else np.zeros((0, 3))

            particle_data = {
                "positions": active_positions,
                "radius": 0.006,
                "color": np.full_like(active_positions, [1.0, 0.3, 0.1]),
            } if len(active_positions) > 0 else None

            nozzle_data = {
                "position": tuple(nozzle_pos),
                "radius": 0.04,
                "color": (0.1, 0.8, 0.2),
            }

            renderer.save_frame(
                wall_data=wall, frame=frame,
                particle_data=particle_data, nozzle_data=nozzle_data,
            )

        if (frame + 1) % 500 == 0 or frame == n_frames - 1:
            coverage = wall.get_coverage()
            elapsed = time.time() - t0
            print(f"    frame {frame + 1}/{n_frames}  "
                  f"coverage={coverage:.1f}%  elapsed={elapsed:.1f}s")

    renderer.finalize()
    coverage = wall.get_coverage()
    print(f"\n[WARP] Final coverage: {coverage:.1f}%")
    return coverage, save_frame_indices


# ═══════════════════════════════════════════════════════════════════
#  Trajectory Plotting
# ═══════════════════════════════════════════════════════════════════
def plot_trajectory(ee_positions_np, warp_positions_np, pen_down_mask, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # cuRobo XZ plane — colour by pen state
    ax = axes[0]
    down = pen_down_mask
    ax.plot(ee_positions_np[down, 0], ee_positions_np[down, 2],
            "g.", ms=0.5, alpha=0.7, label="pen-down")
    ax.plot(ee_positions_np[~down, 0], ee_positions_np[~down, 2],
            "r.", ms=0.2, alpha=0.3, label="pen-up")
    ax.set_xlabel("cuRobo X (m)")
    ax.set_ylabel("cuRobo Z (m)")
    ax.set_title("cuRobo EE Path (XZ wall plane)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Warp XY plane
    ax = axes[1]
    ax.plot(warp_positions_np[down, 0], warp_positions_np[down, 1],
            "g.", ms=0.5, alpha=0.7, label="pen-down")
    ax.plot(warp_positions_np[~down, 0], warp_positions_np[~down, 1],
            "r.", ms=0.2, alpha=0.3, label="pen-up")
    ax.set_xlabel("Warp X (m)")
    ax.set_ylabel("Warp Y (m)")
    ax.set_title("Warp Nozzle Path (XY wall plane)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Trajectory plot: {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  cuRobo + Warp: Spray \"ORIGIN\" Letters (Robot Mesh)")
    print("=" * 65)

    # ── Phase 1a: FK at home ────────────────────────────────────────
    print("\n[Phase 1a] FK at home configuration ...")
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"),
                                   dtype=torch.float32)
    robot_cfg_raw = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg_raw["kinematics"]["urdf_path"] = _URDF_PATH
    robot_cfg_raw["kinematics"]["collision_spheres"] = _SPHERES_PATH

    dummy_world = WorldConfig(cuboid=[
        Cuboid(name="ground", pose=[0, 0, -0.50, 1, 0, 0, 0], dims=[6, 6, 0.02]),
    ])
    tmp_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg_raw, dummy_world, tensor_args,
        num_ik_seeds=4, num_trajopt_seeds=2, interpolation_dt=0.05,
    )
    tmp_mg = MotionGen(tmp_cfg)
    tmp_mg.warmup(enable_graph=False, warmup_js_trajopt=False)

    home_q_t = torch.tensor(HOME_Q, device="cuda:0", dtype=torch.float32)
    hp, hq = compute_fk(tmp_mg, home_q_t)
    hp_np = hp.cpu().numpy()
    hq_np = hq.cpu().numpy()
    print(f"  Home EE: ({hp_np[0]:.4f}, {hp_np[1]:.4f}, {hp_np[2]:.4f})")
    del tmp_mg, tmp_cfg

    # ── Phase 1b: Wall + MotionGen ──────────────────────────────────
    wall_center_y = float(hp_np[1]) + 0.40
    wall_center_z = float(hp_np[2])
    wall_surface_y = wall_center_y - WALL_THICKNESS / 2.0
    ee_y_down = wall_surface_y - SPRAY_STANDOFF
    ee_y_up = wall_surface_y - SPRAY_STANDOFF - PEN_UP_EXTRA

    print(f"\n[Phase 1b] Wall center=(0, {wall_center_y:.3f}, {wall_center_z:.3f})")
    mg, tensor_args = build_motion_gen(wall_center_y, wall_center_z)

    # ── Phase 1c: Letter waypoints ──────────────────────────────────
    spray_quat = hq_np.copy()
    center_x = float(hp_np[0])
    waypoints = _generate_word_waypoints(
        WORD, center_x, ee_y_down, ee_y_up, wall_center_z, spray_quat)

    n_legs = len(waypoints) + 1  # +1 for return-to-home
    print(f"\n[Phase 1c] Writing \"{WORD}\" — {len(waypoints)} waypoints + return home")

    # ── Phase 1d: Dual-strategy planning ────────────────────────────
    #   pen-up  → plan_single (trajectory optimizer, free path)
    #   pen-down → IK at target + linear joint-space interpolation
    #              → guarantees straight-line Cartesian EE motion
    print(f"\n[Phase 1d] Planning {n_legs} legs (pen-up: trajopt, pen-down: IK+linear) ...")

    trajs = []
    pen_flags = []  # True/False per trajectory segment
    current_q = tensor_args.to_device([HOME_Q])
    fail_count = 0

    for wi, (lab, pos, quat, pen) in enumerate(waypoints):
        leg_num = wi + 1
        pen_str = "PEN-DOWN" if pen else "pen-up"
        print(f"  [{leg_num:3d}/{n_legs}] {pen_str:>8s}  -> {lab:20s}  "
              f"({pos[0]:+.3f}, {pos[1]:.3f}, {pos[2]:.3f})", end="  ")

        goal = _make_pose(pos, quat, tensor_args)

        if not pen:
            # ── Pen-up: plan_single (path shape doesn't matter) ──────
            result = mg.plan_single(
                JointState.from_position(current_q, joint_names=JOINT_NAMES),
                goal)
            if result.success.item():
                traj = result.get_interpolated_plan()
                trajs.append(traj)
                pen_flags.append(False)
                current_q = traj.position[-1:].clone()
                print(f"OK  steps={traj.position.shape[0]}")
            else:
                fail_count += 1
                print("FAILED")
        else:
            # ── Pen-down: IK + linear joint interp = straight line ───
            seed = current_q.view(1, 1, -1)
            ik_result = mg.solve_ik(goal, seed_config=seed, return_seeds=1)

            if not ik_result.success.any():
                fail_count += 1
                print("IK FAILED")
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
            pen_flags.append(True)
            current_q = goal_q.clone()
            print(f"IK+linear OK  steps={JOINT_INTERP_STEPS}")

    # ── Return to home ──────────────────────────────────────────────
    print(f"  [{n_legs:3d}/{n_legs}]   pen-up  -> Home", end="  ")
    home_goal = _make_pose(hp_np, hq_np, tensor_args)
    result = mg.plan_single(
        JointState.from_position(current_q, joint_names=JOINT_NAMES),
        home_goal,
    )
    if result.success.item():
        traj = result.get_interpolated_plan()
        trajs.append(traj)
        pen_flags.append(False)
        print(f"OK  steps={traj.position.shape[0]}")
    else:
        fail_count += 1
        print("FAILED")

    if not trajs:
        print("\n[ERROR] No successful legs. Aborting.")
        return

    print(f"\n  Planning: {len(trajs)} OK, {fail_count} failed")

    # ── Phase 2a: Extract EE path + pen_down mask ───────────────────
    print("\n[Phase 2a] Extracting EE trajectory ...")
    all_q = torch.cat([t.position for t in trajs], dim=0)
    total_frames = all_q.shape[0]
    all_q_np = all_q.cpu().numpy()
    print(f"  Total frames: {total_frames}")

    # Build per-frame pen_down mask
    frame_pen_down = np.zeros(total_frames, dtype=bool)
    offset = 0
    for traj, pen in zip(trajs, pen_flags):
        n = traj.position.shape[0]
        if pen:
            frame_pen_down[offset:offset + n] = True
        offset += n
    pen_down_count = int(frame_pen_down.sum())
    print(f"  Pen-down frames: {pen_down_count} / {total_frames}")

    batch_size = 512
    ee_pos_list = []
    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        ee_pos_list.append(compute_fk_batch_ee(mg, all_q[start:end]).cpu())
    ee_positions = torch.cat(ee_pos_list, dim=0).numpy()

    # ── Phase 2b: Coordinate mapping ────────────────────────────────
    print("[Phase 2b] Coordinate mapping ...")
    warp_nozzle_pos = curobo_to_warp(
        ee_positions, wall_surface_y, center_x, wall_center_z)

    print(f"  Nozzle X: [{warp_nozzle_pos[:,0].min():.3f}, {warp_nozzle_pos[:,0].max():.3f}]")
    print(f"  Nozzle Y: [{warp_nozzle_pos[:,1].min():.3f}, {warp_nozzle_pos[:,1].max():.3f}]")
    print(f"  Nozzle Z: [{warp_nozzle_pos[:,2].min():.3f}, {warp_nozzle_pos[:,2].max():.3f}]")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(_OUTPUT_DIR, "spray_trajectory.png")
    plot_trajectory(ee_positions, warp_nozzle_pos, frame_pen_down, plot_path)

    # ── Phase 3: Warp spray simulation ──────────────────────────────
    print("\n[Phase 3] Warp spray simulation ...")
    coverage, save_frame_indices = run_spray_simulation(
        warp_nozzle_pos, frame_pen_down, _OUTPUT_DIR)

    # ── Phase 4: Add robot mesh to USD ──────────────────────────────
    print("\n[Phase 4] Computing FK for all links (yourdfpy) ...")
    saved_q = all_q_np[save_frame_indices]
    link_transforms = compute_all_link_fk(
        saved_q, JOINT_NAMES, _URDF_PATH, VISUAL_LINKS)

    # Transform link poses to Warp coordinates
    print("  Transforming link poses to Warp coordinates ...")
    link_transforms_warp = {}
    for ln, tfs in link_transforms.items():
        tfs_warp = np.zeros_like(tfs)
        for fi in range(len(tfs)):
            tfs_warp[fi] = _transform_4x4_curobo_to_warp(
                tfs[fi], wall_surface_y, center_x, wall_center_z)
        link_transforms_warp[ln] = tfs_warp

    spray_usd = os.path.join(_OUTPUT_DIR, "spray_result.usd")
    print(f"  Adding robot mesh to {spray_usd} ...")
    add_robot_to_usd(
        spray_usd, _ROBOT_USD_PATH,
        link_transforms_warp, save_frame_indices, 60)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Results")
    print("=" * 65)
    print(f"  Word           : {WORD}")
    print(f"  Legs planned   : {len(trajs)} ({fail_count} failed)")
    print(f"  Total frames   : {total_frames}")
    print(f"  Pen-down frames: {pen_down_count}")
    print(f"  Wall coverage  : {coverage:.1f}%")
    print(f"  USD (spray+bot): {spray_usd}")
    print(f"  Trajectory     : {plot_path}")
    print(f"\n  View:  usdview {spray_usd}")
    print("=" * 65)


if __name__ == "__main__":
    main()
