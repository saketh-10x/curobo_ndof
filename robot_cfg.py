"""Configuration for the 10x V1.1.1 mobile manipulator.

The following configuration parameters are available:

* :obj:`ROBOT_V1_1_CFG`: The V1.1.1 mobile manipulator with UR10e arm and sprayer tool.

The robot has the following joint groups:

* Virtual base: ``base_x``, ``base_y``, ``base_theta`` (prismatic x/y + revolute theta)
* Steering: ``steering_f``, ``steering_rl``, ``steering_rr``
* Wheels: ``wheel_f``, ``wheel_rl``, ``wheel_rr`` (continuous)
* Liftkit: ``liftkit_mid`` (prismatic), ``liftkit_top`` (mimic of liftkit_mid)
* UR10e arm: ``shoulder_pan_joint``, ``shoulder_lift_joint``, ``elbow_joint``,
  ``wrist_1_joint``, ``wrist_2_joint``, ``wrist_3_joint``

End-effector link: ``tool0``
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

##
# Paths
##

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "full_robot_resolved.urdf")
_USD_DIR = os.path.join(_SCRIPT_DIR, "usd_generated")

##
# Configuration
##

ROBOT_V1_1_CFG = ArticulationCfg(
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,  # -pi/2
            "elbow_joint": 1.5708,  # pi/2
            "wrist_1_joint": -1.5708,  # -pi/2
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        # Virtual base joints — high damping to prevent drift (ridgeback pattern)
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_x", "base_y", "base_theta"],
            effort_limit=1000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        # UR10e shoulder joints -- gains from joint_config_sim.yaml
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=20000.0,
            damping=2000.0,
            effort_limit=19800.0,
            friction=0.0,
            armature=0.0,
        ),
        # UR10e elbow joint
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=20000.0,
            damping=2000.0,
            effort_limit=19800.0,
            friction=0.0,
            armature=0.0,
        ),
        # UR10e wrist joints
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=20000.0,
            damping=2000.0,
            effort_limit=19800.0,
            friction=0.0,
            armature=0.0,
        ),
        # Liftkit prismatic joints -- very high stiffness to hold in place
        "liftkit": ImplicitActuatorCfg(
            joint_names_expr=["liftkit_.*"],
            stiffness=8000000.0,
            damping=45000.0,
            effort_limit=70000.0,
            velocity_limit=0.012,
            friction=0.0,
            armature=0.0,
        ),
        # Swerve steering joints
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_.*"],
            stiffness=1000.0,
            damping=100.0,
            effort_limit=0.5,
            velocity_limit=120.0,
            friction=0.0,
            armature=0.0,
        ),
        # Swerve wheel joints (continuous)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"],
            stiffness=300.0,
            damping=20000.0,
            effort_limit=2.0,
            velocity_limit=100.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of the 10x V1.1.1 mobile manipulator with UR10e arm.

The following control configuration is used:

* Arm (shoulder/elbow/wrist): position control with high stiffness (20000) and damping (2000)
* Liftkit: position control with very high stiffness (8M) to hold in place
* Steering: position control with moderate gains
* Wheels: position control with high damping for velocity-like behavior
"""
