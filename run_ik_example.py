"""cuRobo IK solver example for V1.1.1 robot with Isaac Sim execution.

Solves IK for a target EE pose using cuRobo's IKSolver, then applies
the solved joint positions to the simulated robot.

Usage:
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p scripts/standalone/curobo_v1_1/run_ik_example.py --headless
    # With GUI:
    OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -p scripts/standalone/curobo_v1_1/run_ik_example.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cuRobo IK example for V1.1.1 robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

# cuRobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# Reuse robot scene config from existing codebase
from robot_cfg import ROBOT_V1_1_CFG

##
# Paths
##
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")

# 7-DOF joint names (must match YAML cspace)
JOINT_NAMES = [
    "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Virtual base joints
BASE_JOINT_NAMES = ["base_x", "base_y", "base_theta"]


##
# Scene Configuration
##
@configclass
class IKSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, light, and V1.1.1 robot."""

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
# Helpers
##

def build_ik_solver():
    """Build cuRobo IKSolver from the robot YAML config."""
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    # Load robot config from YAML (same as MotionGen path)
    config_data = load_yaml(_ROBOT_CFG_PATH)
    robot_cfg_dict = config_data["robot_cfg"]
    robot_cfg_dict["kinematics"]["urdf_path"] = _URDF_PATH

    robot_cfg = RobotConfig.from_dict(robot_cfg_dict)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_model=None,  # No obstacles for this example
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    print("[INFO]: cuRobo IKSolver built successfully.")
    return ik_solver, tensor_args


def get_sim_joint_indices(robot, names):
    """Get sim joint indices for a list of joint names."""
    sim_names = list(robot.data.joint_names)
    return [sim_names.index(n) for n in names]


def get_base_frame_w(robot):
    """Get articulation root (swerve_link_x) pose in world."""
    return robot.data.root_pose_w[0:1, 0:3], robot.data.root_pose_w[0:1, 3:7]


def get_tool0_pose_w(robot):
    """Get tool0 body pose in world frame."""
    body_names = list(robot.data.body_names)
    tool0_idx = body_names.index("tool0")
    return (
        robot.data.body_state_w[0:1, tool0_idx, 0:3],
        robot.data.body_state_w[0:1, tool0_idx, 3:7],
    )


def get_tool0_pose_in_base(robot):
    """Get tool0 pose relative to swerve_link_x (kinematic base)."""
    base_pos_w, base_quat_w = get_base_frame_w(robot)
    tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot)
    return subtract_frame_transforms(base_pos_w, base_quat_w, tool0_pos_w, tool0_quat_w)


def sim_step_n(sim, scene, n):
    """Run n simulation steps."""
    dt = sim.get_physics_dt()
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def quat_angular_error(q1, q2):
    """Compute angular error in radians between two wxyz quaternions."""
    q1 = q1 / q1.norm()
    q2 = q2 / q2.norm()
    dot = torch.abs(torch.dot(q1, q2)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def apply_joint_positions_to_sim(robot, sim, scene, joint_positions, steps=200):
    """Apply 7-DOF joint positions to the sim robot and settle.

    Args:
        robot: The articulation robot from the scene.
        sim: The simulation context.
        scene: The interactive scene.
        joint_positions: Tensor of shape (7,) with cuRobo joint order.
        steps: Number of sim steps to settle.
    """
    joint_sim_indices = get_sim_joint_indices(robot, JOINT_NAMES)
    sim_joint_names = list(robot.data.joint_names)
    liftkit_mid_idx = sim_joint_names.index("liftkit_mid")
    liftkit_top_idx = sim_joint_names.index("liftkit_top")

    # Build full joint target from current state, overwrite the 7 controlled joints
    full_target = robot.data.joint_pos[0, :].clone()
    for ci, si in enumerate(joint_sim_indices):
        full_target[si] = joint_positions[ci].to(full_target.device)

    # Enforce liftkit_top = liftkit_mid (mimic joint)
    full_target[liftkit_top_idx] = full_target[liftkit_mid_idx]

    robot.set_joint_position_target(full_target.unsqueeze(0))

    # Let PD controller settle
    sim_step_n(sim, scene, steps)


##
# Main
##

def main():
    # -- Setup sim + scene --
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 1.0])

    scene_cfg = IKSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Scene setup complete.")

    robot = scene["robot"]

    # -- Build cuRobo IK solver --
    ik_solver, tensor_args = build_ik_solver()

    # -- Settle robot after spawn --
    print("[INFO]: Settling robot (100 steps)...")
    sim_step_n(sim, scene, 100)

    # -- Create visualization markers --
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    goal_frame_cfg = FRAME_MARKER_CFG.copy()
    goal_frame_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    goal_marker = VisualizationMarkers(goal_frame_cfg.replace(prim_path="/Visuals/ee_goal"))

    # =========================================================================
    # EXAMPLE 1: Solve IK for random sampled poses (cuRobo's FK -> IK round-trip)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 1: IK solve for random FK-sampled poses")
    print("=" * 70)

    for i in range(3):
        # Sample a random valid joint config, compute its FK to get a reachable goal
        q_sample = ik_solver.sample_configs(1)
        kin_state = ik_solver.fk(q_sample)
        goal = CuroboPose(kin_state.ee_position, kin_state.ee_quaternion)

        print(f"\n--- Sample {i+1} ---")
        print(f"  Sampled q:   {q_sample[0].tolist()}")
        print(f"  FK goal pos: {kin_state.ee_position[0].tolist()}")
        print(f"  FK goal quat:{kin_state.ee_quaternion[0].tolist()}")

        # Solve IK
        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        solve_time = time.time() - st_time

        print(f"  IK success:  {result.success.item()}")
        print(f"  Solve time:  {solve_time:.4f} s")
        print(f"  Pos error:   {result.position_error.item() * 1000:.2f} mm")
        print(f"  Rot error:   {math.degrees(result.rotation_error.item()):.2f} deg")

        if result.success.item():
            # Get the solved joint positions
            solved_q = result.solution[0, 0, :]  # (num_seeds, batch, dof) -> first solution
            print(f"  Solved q:    {solved_q.tolist()}")

            # Apply to sim
            print(f"  Applying to sim (200 steps)...")
            apply_joint_positions_to_sim(robot, sim, scene, solved_q, steps=200)

            # Verify in sim: get tool0 in base frame
            sim_pos, sim_quat = get_tool0_pose_in_base(robot)
            goal_pos = kin_state.ee_position[0]
            goal_quat = kin_state.ee_quaternion[0]

            pos_err = torch.norm(sim_pos[0].to(goal_pos.device) - goal_pos).item()
            quat_err = quat_angular_error(
                sim_quat[0].to(goal_quat.device), goal_quat
            ).item()

            print(f"  Sim EE pos:  {sim_pos[0].tolist()}")
            print(f"  Goal pos:    {goal_pos.tolist()}")
            print(f"  Sim pos err: {pos_err * 1000:.1f} mm")
            print(f"  Sim rot err: {math.degrees(quat_err):.2f} deg")

            ok = pos_err < 0.020 and quat_err < math.radians(10.0)
            print(f"  Verification: {'PASS' if ok else 'FAIL'}")

            # Update markers
            base_pos_w, base_quat_w = get_base_frame_w(robot)
            goal_pos_w, goal_quat_w = combine_frame_transforms(
                base_pos_w, base_quat_w,
                goal_pos.unsqueeze(0).to(base_pos_w.device),
                goal_quat.unsqueeze(0).to(base_quat_w.device),
            )
            tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot)
            ee_marker.visualize(tool0_pos_w, tool0_quat_w)
            goal_marker.visualize(goal_pos_w, goal_quat_w)

    # =========================================================================
    # EXAMPLE 2: Solve IK for specific target poses and apply to sim
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: IK solve for specific target poses")
    print("=" * 70)

    # Define target poses in base frame (swerve_link_x frame)
    # These are positions the tool0 should reach
    target_poses = [
        {
            "name": "Front reach",
            "pos": [0.8, 0.0, 1.2],   # x forward, z up from base
            "quat": [0.0, 0.707, 0.0, 0.707],  # wxyz: pointing downward-ish
        },
        {
            "name": "Side reach",
            "pos": [0.0, 0.6, 1.0],
            "quat": [0.5, 0.5, 0.5, 0.5],  # wxyz
        },
        {
            "name": "High up",
            "pos": [0.3, 0.0, 1.5],
            "quat": [1.0, 0.0, 0.0, 0.0],  # wxyz: identity
        },
    ]

    for target in target_poses:
        print(f"\n--- Target: {target['name']} ---")

        goal_pos = torch.tensor(
            [target["pos"]], device=tensor_args.device, dtype=tensor_args.dtype
        )
        goal_quat = torch.tensor(
            [target["quat"]], device=tensor_args.device, dtype=tensor_args.dtype
        )
        # Normalize quaternion
        goal_quat = goal_quat / goal_quat.norm(dim=-1, keepdim=True)

        goal = CuroboPose(goal_pos, goal_quat)

        print(f"  Goal pos:  {goal_pos[0].tolist()}")
        print(f"  Goal quat: {goal_quat[0].tolist()}")

        # Solve IK
        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        solve_time = time.time() - st_time

        print(f"  IK success:  {result.success.item()}")
        print(f"  Solve time:  {solve_time:.4f} s")
        print(f"  Pos error:   {result.position_error.item() * 1000:.2f} mm")
        print(f"  Rot error:   {math.degrees(result.rotation_error.item()):.2f} deg")

        if result.success.item():
            solved_q = result.solution[0, 0, :]
            print(f"  Solved q:    {solved_q.tolist()}")

            # Apply to sim
            print(f"  Applying to sim (300 steps for settle)...")
            apply_joint_positions_to_sim(robot, sim, scene, solved_q, steps=300)

            # Verify in sim
            sim_pos, sim_quat = get_tool0_pose_in_base(robot)

            pos_err = torch.norm(
                sim_pos[0].to(goal_pos.device) - goal_pos[0]
            ).item()
            quat_err = quat_angular_error(
                sim_quat[0].to(goal_quat.device), goal_quat[0]
            ).item()

            print(f"  Sim EE pos:  {sim_pos[0].tolist()}")
            print(f"  Sim pos err: {pos_err * 1000:.1f} mm")
            print(f"  Sim rot err: {math.degrees(quat_err):.2f} deg")

            ok = pos_err < 0.020 and quat_err < math.radians(10.0)
            print(f"  Verification: {'PASS' if ok else 'FAIL'}")

            # Update markers
            base_pos_w, base_quat_w = get_base_frame_w(robot)
            goal_pos_w, goal_quat_w = combine_frame_transforms(
                base_pos_w, base_quat_w,
                goal_pos.to(base_pos_w.device),
                goal_quat.to(base_quat_w.device),
            )
            tool0_pos_w, tool0_quat_w = get_tool0_pose_w(robot)
            ee_marker.visualize(tool0_pos_w, tool0_quat_w)
            goal_marker.visualize(goal_pos_w, goal_quat_w)
        else:
            print(f"  IK failed — target may be unreachable. Skipping sim apply.")

    # =========================================================================
    # EXAMPLE 3: Batch IK — solve multiple goals at once
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Batch IK (10 random goals solved simultaneously)")
    print("=" * 70)

    num_goals = 10
    q_samples = ik_solver.sample_configs(num_goals)
    kin_states = ik_solver.fk(q_samples)
    batch_goal = CuroboPose(kin_states.ee_position, kin_states.ee_quaternion)

    st_time = time.time()
    batch_result = ik_solver.solve_batch(batch_goal)
    torch.cuda.synchronize()
    solve_time = time.time() - st_time

    success_count = torch.count_nonzero(batch_result.success).item()
    print(f"  Batch size:    {num_goals}")
    print(f"  Successes:     {success_count}/{num_goals}")
    print(f"  Success rate:  {success_count / num_goals * 100:.0f}%")
    print(f"  Solve time:    {solve_time:.4f} s ({num_goals / solve_time:.0f} Hz)")
    print(f"  Mean pos err:  {batch_result.position_error.mean().item() * 1000:.2f} mm")
    print(f"  Mean rot err:  {math.degrees(batch_result.rotation_error.mean().item()):.2f} deg")

    # Apply the LAST successful solution to sim as a demo
    success_mask = batch_result.success.view(-1)
    if success_mask.any():
        last_idx = torch.where(success_mask)[0][-1].item()
        solved_q = batch_result.solution[0, last_idx, :]
        goal_pos = kin_states.ee_position[last_idx]
        goal_quat = kin_states.ee_quaternion[last_idx]

        print(f"\n  Applying last successful solution (goal #{last_idx}) to sim...")
        print(f"  Goal pos:  {goal_pos.tolist()}")
        print(f"  Solved q:  {solved_q.tolist()}")

        apply_joint_positions_to_sim(robot, sim, scene, solved_q, steps=300)

        sim_pos, sim_quat = get_tool0_pose_in_base(robot)
        pos_err = torch.norm(sim_pos[0].to(goal_pos.device) - goal_pos).item()
        print(f"  Sim pos err: {pos_err * 1000:.1f} mm")
        print(f"  Verification: {'PASS' if pos_err < 0.020 else 'FAIL'}")

    # -- Done --
    print("\n" + "=" * 70)
    print("IK EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
