"""Follow a movable cube with the V1.1.1 robot arm using Differential IK.

Spawns the robot and a target cube. The robot arm tracks the cube position
in real-time using IsaacLab's DifferentialIKController. Move the cube in the
Isaac Sim viewport using the transform gizmo (select the cube, press W/E/R).

Only the 6-DOF UR10e arm joints are controlled. Base, liftkit, steering,
and wheels remain passive.

.. code-block:: bash

    # With GUI (required for interactive cube movement):
    ./isaaclab.sh -p scripts/standalone/curobo_v1_1/follow_cube.py

    # Headless (cube moves on a scripted path):
    ./isaaclab.sh -p scripts/standalone/curobo_v1_1/follow_cube.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Follow cube with V1.1.1 robot arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
# isort: off
from robot_cfg import ROBOT_V1_1_CFG

# isort: on


##
# Scene
##


class FollowCubeSceneCfg(InteractiveSceneCfg):
    """Scene with robot and a movable target cube."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = ROBOT_V1_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Target cube -- kinematic rigid body (no gravity, user can move it in viewport)
    target_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 1.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


##
# Simulation loop
##


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with differential IK cube tracking."""

    robot: Articulation = scene["robot"]
    target_cube: RigidObject = scene["target_cube"]

    # -- Configure which joints to control (arm only) --
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        body_names=["tool0"],
    )
    robot_entity_cfg.resolve(scene)

    # For floating-base robots, Jacobian body index = body_id (no -1 offset)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    print(f"[INFO]: Robot is_fixed_base: {robot.is_fixed_base}")
    print(f"[INFO]: EE body id: {robot_entity_cfg.body_ids[0]}, Jacobian idx: {ee_jacobi_idx}")
    print(f"[INFO]: Arm joint ids: {robot_entity_cfg.joint_ids}")
    print(f"[INFO]: Arm joint names: {[robot.data.joint_names[i] for i in robot_entity_cfg.joint_ids]}")

    # -- Create Differential IK controller --
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # -- Visualization markers --
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # -- Buffers --
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
    joint_pos_des = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()

    sim_dt = sim.get_physics_dt()
    count = 0
    settling_steps = 50

    print("[INFO]: Starting follow-cube loop.")
    print("[INFO]: Move the green cube in the viewport (select it, press W to translate).")

    while simulation_app.is_running():
        # -- Initial settling --
        if count < settling_steps:
            scene.write_data_to_sim()
            sim.step()
            count += 1
            scene.update(sim_dt)
            continue

        # -- Reset periodically (only joint state, not cube) --
        if count % 1000 == 0 and count > settling_steps:
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            diff_ik_controller.reset()
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            print("[INFO]: Reset robot joint state.")

        # -- Read cube world pose --
        cube_pose_w = target_cube.data.root_pose_w  # (num_envs, 7)

        # -- Convert cube pose from world frame to robot root frame --
        root_pose_w = robot.data.root_pose_w  # (num_envs, 7)
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            cube_pose_w[:, 0:3], cube_pose_w[:, 3:7],
        )

        # -- Set IK command (target pose in root frame) --
        ik_commands[:, 0:3] = cube_pos_b
        ik_commands[:, 3:7] = cube_quat_b
        diff_ik_controller.set_command(ik_commands)

        # -- Compute Differential IK --
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # -- Apply joint position targets (arm only) --
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # -- Step simulation --
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # -- Update markers --
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(cube_pose_w[:, 0:3], cube_pose_w[:, 3:7])

        # -- Print tracking error periodically --
        if count % 200 == 0:
            pos_err = torch.norm(ee_pose_w[:, 0:3] - cube_pose_w[:, 0:3], dim=-1)
            print(f"[INFO]: Step {count}, tracking error: {pos_err[0].item()*1000:.1f} mm")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = FollowCubeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
