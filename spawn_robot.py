"""Spawn the V1.1.1 robot in an Isaac Lab environment.

.. code-block:: bash

    # With GUI:
    ./isaaclab.sh -p scripts/standalone/curobo_v1_1/spawn_robot.py

    # Headless:
    ./isaaclab.sh -p scripts/standalone/curobo_v1_1/spawn_robot.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn V1.1.1 robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

##
# Pre-defined configs
##
# isort: off
from robot_cfg import ROBOT_V1_1_CFG

# isort: on


class RobotSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, light, and V1.1.1 robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = ROBOT_V1_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    count = 0

    robot: Articulation = scene["robot"]

    while simulation_app.is_running():
        # Reset periodically
        if count % 500 == 0:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot: Articulation = scene["robot"]
    print(f"[INFO]: Robot spawned. Joint names: {robot.data.joint_names}")
    print(f"[INFO]: Body names: {robot.data.body_names}")
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
