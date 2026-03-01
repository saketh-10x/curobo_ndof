"""Test cuRobo IK solutions for V1.1.1 mobile manipulator (arm-only mode).

Verifies that cuRobo returns valid joint solutions for the 6-DOF UR10e arm
given target end-effector poses specified as (x, y, z, a, b, c) where a, b, c
are roll, pitch, yaw Euler angles in radians.

Mobile base and liftkit are locked at zero in the YAML config.

This is a pure cuRobo test - no Isaac Sim required.

Usage:
    ./isaaclab.sh -p scripts/standalone/curobo_v1_1/test_curobo_ik.py
"""

import os
import sys
import math
import torch
import numpy as np

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.state import JointState as CuroboJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.util_file import load_yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def setup_curobo():
    """Initialize cuRobo MotionGen."""
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH

    world_cfg = WorldConfig()

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args=tensor_args,
        collision_checker_type=None,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"mesh": 0, "obb": 0},
        trajopt_tsteps=32,
        collision_activation_distance=0.02,
        self_collision_check=False,
        position_threshold=0.005,
        rotation_threshold=0.05,
    )

    motion_gen = MotionGen(motion_gen_config)
    print("[INFO]: Warming up cuRobo MotionGen...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[INFO]: Warmup complete.")

    plan_config = MotionGenPlanConfig(
        enable_graph=True,
        enable_graph_attempt=4,
        max_attempts=10,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    return motion_gen, plan_config, tensor_args


def test_ik_for_pose(motion_gen, plan_config, tensor_args, x, y, z, roll, pitch, yaw, start_config=None):
    """Test IK solution for a given xyzabc target.

    Returns:
        dict with success, joint_positions (per DOF), ee_error_pos, ee_error_rot
    """
    quat = euler_to_quaternion(roll, pitch, yaw)

    curobo_joint_names = motion_gen.kinematics.joint_names
    num_dof = len(curobo_joint_names)

    # Start state (retract config from YAML)
    if start_config is None:
        start_config = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]  # 6-DOF arm retract

    start_pos = torch.tensor([start_config[:num_dof]], device=tensor_args.device, dtype=tensor_args.dtype)
    start_vel = torch.zeros_like(start_pos)
    start_acc = torch.zeros_like(start_pos)

    start_state = CuroboJointState(
        position=start_pos,
        velocity=start_vel,
        acceleration=start_acc,
        joint_names=curobo_joint_names,
        tensor_args=tensor_args,
    )

    goal_position = torch.tensor([[x, y, z]], device=tensor_args.device, dtype=tensor_args.dtype)
    goal_quaternion = torch.tensor([quat], device=tensor_args.device, dtype=tensor_args.dtype)
    goal_pose = CuroboPose(position=goal_position, quaternion=goal_quaternion)

    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    if result.success.item():
        interpolated = result.get_interpolated_plan()
        full_plan = motion_gen.get_full_js(interpolated)
        common_names = [n for n in curobo_joint_names if n in full_plan.joint_names]
        full_plan = full_plan.get_ordered_joint_state(common_names)

        final_js = full_plan.position[-1]

        # Compute FK to verify EE pose
        final_state = CuroboJointState(
            position=final_js.unsqueeze(0),
            velocity=torch.zeros(1, num_dof, device=tensor_args.device, dtype=tensor_args.dtype),
            acceleration=torch.zeros(1, num_dof, device=tensor_args.device, dtype=tensor_args.dtype),
            joint_names=curobo_joint_names,
            tensor_args=tensor_args,
        )

        kin_result = motion_gen.compute_kinematics(final_state)
        ee_pos = kin_result.ee_pose.position.squeeze().cpu().numpy()
        ee_quat = kin_result.ee_pose.quaternion.squeeze().cpu().numpy()

        pos_error = np.linalg.norm(ee_pos - np.array([x, y, z]))
        quat_target = np.array(quat)
        quat_dot = abs(np.dot(ee_quat, quat_target))
        rot_error = 2.0 * np.arccos(min(quat_dot, 1.0))

        joint_values = {}
        for i, name in enumerate(curobo_joint_names):
            joint_values[name] = final_js[i].item()

        return {
            "success": True,
            "joint_values": joint_values,
            "ee_pos_achieved": ee_pos.tolist(),
            "ee_quat_achieved": ee_quat.tolist(),
            "pos_error_m": pos_error,
            "rot_error_rad": rot_error,
            "num_waypoints": len(full_plan.position),
        }
    else:
        return {
            "success": False,
            "status": str(result.status),
            "joint_values": {},
            "pos_error_m": None,
            "rot_error_rad": None,
        }


def main():
    motion_gen, plan_config, tensor_args = setup_curobo()

    curobo_joint_names = motion_gen.kinematics.joint_names
    print(f"\n{'='*80}")
    print(f"cuRobo Kinematic Chain Info")
    print(f"{'='*80}")
    print(f"Joint names ({len(curobo_joint_names)} DOF): {curobo_joint_names}")
    print(f"EE link: {motion_gen.kinematics.ee_link}")
    print(f"Base link: {motion_gen.kinematics.base_link}")

    # Test poses within arm-only workspace (base and liftkit locked at 0)
    # The arm is mounted on the robot chassis ~1m above ground
    # UR10e reach is ~1.3m, tool0 adds some offset
    test_poses = [
        {
            "name": "Arm: Forward reach",
            "xyzabc": (0.5, 0.0, 1.5, 0.0, math.pi / 2, 0.0),
        },
        {
            "name": "Arm: Side reach",
            "xyzabc": (0.0, 0.5, 1.5, 0.0, 0.0, math.pi / 2),
        },
        {
            "name": "Arm: Above",
            "xyzabc": (-0.2, 0.0, 2.0, 0.0, 0.0, 0.0),
        },
        {
            "name": "Arm: Low forward",
            "xyzabc": (0.6, 0.0, 1.0, 0.0, math.pi / 2, 0.0),
        },
        {
            "name": "Arm: Rotated diagonal",
            "xyzabc": (0.4, 0.4, 1.5, math.pi / 4, math.pi / 4, 0.0),
        },
        {
            "name": "Arm: Behind and up",
            "xyzabc": (-0.3, 0.3, 1.8, 0.0, -math.pi / 4, math.pi / 4),
        },
    ]

    print(f"\n{'='*80}")
    print(f"Testing IK Solutions for {len(test_poses)} Target Poses (arm only, base+liftkit locked)")
    print(f"{'='*80}")

    retract = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
    results_summary = []
    for i, test in enumerate(test_poses):
        x, y, z, r, p, yaw = test["xyzabc"]
        print(f"\n--- Test {i+1}: {test['name']} ---")
        print(f"  Target: xyz=({x:.3f}, {y:.3f}, {z:.3f}), rpy=({math.degrees(r):.1f}, {math.degrees(p):.1f}, {math.degrees(yaw):.1f}) deg")

        result = test_ik_for_pose(motion_gen, plan_config, tensor_args, x, y, z, r, p, yaw)

        if result["success"]:
            print(f"  SUCCESS! Waypoints: {result['num_waypoints']}")
            print(f"  Position error: {result['pos_error_m']*1000:.2f} mm")
            print(f"  Rotation error: {math.degrees(result['rot_error_rad']):.2f} deg")
            print(f"  Joint solution ({len(curobo_joint_names)} DOF):")

            jv = result["joint_values"]
            print(f"    shoulder_pan={math.degrees(jv.get('shoulder_pan_joint', 0)):.2f}deg, shoulder_lift={math.degrees(jv.get('shoulder_lift_joint', 0)):.2f}deg")
            print(f"    elbow={math.degrees(jv.get('elbow_joint', 0)):.2f}deg")
            print(f"    wrist_1={math.degrees(jv.get('wrist_1_joint', 0)):.2f}deg, wrist_2={math.degrees(jv.get('wrist_2_joint', 0)):.2f}deg, wrist_3={math.degrees(jv.get('wrist_3_joint', 0)):.2f}deg")

            # Check which DOFs were actually used (changed from retract)
            dof_names = list(jv.keys())
            used_dofs = []
            for j, name in enumerate(dof_names):
                if abs(jv[name] - retract[j]) > 0.01:
                    used_dofs.append(name)
            print(f"    Active DOFs (changed from retract): {used_dofs}")

            results_summary.append({"name": test["name"], "success": True, "pos_err_mm": result["pos_error_m"]*1000, "rot_err_deg": math.degrees(result["rot_error_rad"])})
        else:
            print(f"  FAILED: {result['status']}")
            results_summary.append({"name": test["name"], "success": False})

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Test':<45} {'Status':<10} {'Pos Err':<12} {'Rot Err':<12}")
    print(f"{'-'*45} {'-'*10} {'-'*12} {'-'*12}")
    for r in results_summary:
        status = "PASS" if r["success"] else "FAIL"
        pos = f"{r.get('pos_err_mm', 'N/A'):.2f} mm" if r["success"] else "N/A"
        rot = f"{r.get('rot_err_deg', 'N/A'):.2f} deg" if r["success"] else "N/A"
        print(f"{r['name']:<45} {status:<10} {pos:<12} {rot:<12}")

    passed = sum(1 for r in results_summary if r["success"])
    print(f"\n{passed}/{len(results_summary)} tests passed.")

    if passed < len(results_summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
