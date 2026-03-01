"""Test cuRobo 7-DOF IK planning (liftkit + arm) with multiple goal poses.

No Isaac Sim required — pure cuRobo planning + FK verification.

Usage:
    python scripts/standalone/curobo_v1_1/test_curobo_7dof.py
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.util_file import load_yaml


# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_ROBOT_CFG_PATH = os.path.join(_SCRIPT_DIR, "robotv1_1_sprayer.yml")

# 7-DOF joint names (must match YAML cspace)
JOINT_NAMES = [
    "liftkit_mid",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Joint weights: liftkit expensive, arm cheap
JOINT_WEIGHTS = [8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# ── Result container (follows NdofCuroboPlanner pattern) ──────────────────────

@dataclass
class PlanResult:
    success: bool
    joint_names: List[str]
    positions: Optional[torch.Tensor]       # (T, 7)
    velocities: Optional[torch.Tensor]      # (T, 7)
    accelerations: Optional[torch.Tensor]   # (T, 7)
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


# ── Planner setup ─────────────────────────────────────────────────────────────

def build_motion_gen():
    """Build and warm up cuRobo MotionGen from YAML config."""
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    robot_cfg = load_yaml(_ROBOT_CFG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["urdf_path"] = _URDF_PATH

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=WorldConfig(),
        tensor_args=tensor_args,
        collision_checker_type=None,
        num_ik_seeds=30,
        num_graph_seeds=12,
        num_trajopt_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"mesh": 0, "obb": 0},
        trajopt_tsteps=32,
        collision_activation_distance=0.02,
        self_collision_check=False,
        position_threshold=0.005,
        rotation_threshold=0.05,
    )

    motion_gen = MotionGen(mg_cfg)
    print("[INFO]: Warming up cuRobo MotionGen...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[INFO]: Warmup complete.\n")

    return motion_gen, tensor_args


def plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat,
                 interpolation_dt=0.05, max_attempts=20):
    """Plan a trajectory from current_q to goal pose. Returns PlanResult."""
    ta = tensor_args

    # Ensure 2D
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
        max_attempts=max_attempts,
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
            dt=interpolation_dt, solve_time_s=solve_time,
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
        dt=interpolation_dt, solve_time_s=solve_time,
        motion_time_s=motion_time, status="success",
    )


def compute_fk(motion_gen, q):
    """Compute FK for joint positions q. Returns (ee_pos, ee_quat) as lists."""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    fk = motion_gen.kinematics.get_state(q)
    return fk.ee_position[0].tolist(), fk.ee_quaternion[0].tolist()


# ── Test cases ────────────────────────────────────────────────────────────────

def main():
    motion_gen, tensor_args = build_motion_gen()
    ta = tensor_args

    # Get retract config FK as reference
    retract_q = torch.tensor([[0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]],
                             device=ta.device, dtype=ta.dtype)
    retract_ee_pos, retract_ee_quat = compute_fk(motion_gen, retract_q)
    print(f"Retract config FK:")
    print(f"  joints: {retract_q[0].tolist()}")
    print(f"  EE pos (base_link frame): {retract_ee_pos}")
    print(f"  EE quat: {retract_ee_quat}\n")

    # Get zero config FK
    zero_q = torch.zeros(1, 7, device=ta.device, dtype=ta.dtype)
    zero_ee_pos, zero_ee_quat = compute_fk(motion_gen, zero_q)
    print(f"Zero config FK:")
    print(f"  joints: {zero_q[0].tolist()}")
    print(f"  EE pos (base_link frame): {zero_ee_pos}")
    print(f"  EE quat: {zero_ee_quat}\n")

    # ── Define test goals ─────────────────────────────────────────────────────
    # Strategy: use FK of known joint configs to get guaranteed-reachable poses,
    # plus some manually-chosen poses near the workspace center.

    # Test configs: different arm poses to get reachable EE positions
    test_joint_configs = [
        # [liftkit, pan, lift, elbow, w1, w2, w3]
        [0.0, 0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0],      # Arm folded up
        [0.0, 1.0, -1.0, 0.5, -1.0, 0.0, 0.0],                # Arm to the side
        [0.0, -0.5, -1.2, 1.0, -1.5, 0.5, 0.0],               # Arm diagonal
        [0.3, 0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0],       # Liftkit raised
        [0.5, 0.5, -1.0, 0.8, -1.2, 0.3, 0.0],                # Liftkit + arm combo
    ]

    # Compute FK for each to get goal poses
    test_goals = []
    for i, jc in enumerate(test_joint_configs):
        q = torch.tensor([jc], device=ta.device, dtype=ta.dtype)
        pos, quat = compute_fk(motion_gen, q)
        test_goals.append({
            "name": f"FK-config-{i}",
            "pos": pos,
            "quat": quat,
            "source_joints": jc,
        })
        print(f"Test goal {i} (from joints {jc}):")
        print(f"  EE pos: {pos}, quat: {quat}")

    # Add some offset goals from retract position
    offsets = [
        ([0.15, 0.1, 0.0], "retract+x+y"),
        ([-0.1, 0.0, 0.1], "retract-x+z"),
        ([0.0, -0.15, -0.05], "retract-y-z"),
    ]
    for offset, name in offsets:
        pos = [retract_ee_pos[j] + offset[j] for j in range(3)]
        test_goals.append({
            "name": name,
            "pos": pos,
            "quat": retract_ee_quat,
        })
        print(f"Test goal '{name}': EE pos: {pos}")

    print(f"\n{'='*70}")
    print(f"Running {len(test_goals)} planning tests from retract config")
    print(f"{'='*70}\n")

    # ── Run planning tests ────────────────────────────────────────────────────
    start_q = retract_q.squeeze(0)  # Plan from retract config
    passed = 0
    failed = 0

    for i, goal in enumerate(test_goals):
        goal_pos = torch.tensor(goal["pos"], device=ta.device, dtype=ta.dtype)
        goal_quat = torch.tensor(goal["quat"], device=ta.device, dtype=ta.dtype)

        result = plan_to_pose(motion_gen, tensor_args, start_q, goal_pos, goal_quat)

        if result.success:
            # Verify FK of final waypoint matches goal
            final_q = result.positions[-1].unsqueeze(0)
            final_pos, final_quat = compute_fk(motion_gen, final_q)
            pos_err = sum((a - b)**2 for a, b in zip(final_pos, goal["pos"]))**0.5
            print(f"  [{i}] PASS  '{goal['name']}' — {result}")
            print(f"         goal pos:  {goal['pos']}")
            print(f"         final pos: {final_pos}  (err: {pos_err*1000:.1f} mm)")
            print(f"         final joints: {final_q[0].tolist()}")
            if result.lift_trajectory is not None:
                print(f"         liftkit: {result.lift_trajectory[0].item():.4f} -> {result.lift_trajectory[-1].item():.4f}")
            passed += 1
        else:
            print(f"  [{i}] FAIL  '{goal['name']}' — {result}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*70}")

    # ── Chain planning test: plan from end of one trajectory to next goal ─────
    print(f"\n{'='*70}")
    print(f"Chain planning test: sequential goals from retract")
    print(f"{'='*70}\n")

    current_q = start_q.clone()
    chain_passed = 0
    for i, goal in enumerate(test_goals[:3]):
        goal_pos = torch.tensor(goal["pos"], device=ta.device, dtype=ta.dtype)
        goal_quat = torch.tensor(goal["quat"], device=ta.device, dtype=ta.dtype)

        result = plan_to_pose(motion_gen, tensor_args, current_q, goal_pos, goal_quat)
        if result.success:
            current_q = result.positions[-1]  # Use final waypoint as next start
            print(f"  Chain [{i}] PASS '{goal['name']}' — {result.num_waypoints} wpts")
            chain_passed += 1
        else:
            print(f"  Chain [{i}] FAIL '{goal['name']}' — {result.status}")

    print(f"\nChain: {chain_passed}/3 passed")


if __name__ == "__main__":
    main()
