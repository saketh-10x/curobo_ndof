#!/usr/bin/env python3
"""
Swerve drive pose controller — closed-loop PID control to reach (x, y, theta) targets.

Subscribes:
    /pose_target  (geometry_msgs/Pose2D)  — desired base pose in world frame
    /odom_sim     (nav_msgs/Odometry)     — current pose feedback

Publishes:
    /amr/wheel_pose_control (std_msgs/Float32MultiArray)
        [FW_travel, FW_steer, RL_travel, RL_steer, RR_travel, RR_steer]
        Travel in m/s, steer in degrees.

Usage:
    python3 isaac_sim/teleop/swerve_pose_controller.py
    ros2 topic pub /pose_target geometry_msgs/msg/Pose2D "{x: 1.0, y: 0.5, theta: 1.57}" --once
"""

import math
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PIDState:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    integral_max: float = 1.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


@dataclass
class WheelPosition:
    x: float = 0.0
    y: float = 0.0


@dataclass
class SwerveGeometry:
    """Wheel positions relative to base_link (metres)."""

    front: WheelPosition = field(default_factory=lambda: WheelPosition(0.3225, 0.0))
    rear_left: WheelPosition = field(default_factory=lambda: WheelPosition(-0.3225, 0.245))
    rear_right: WheelPosition = field(default_factory=lambda: WheelPosition(-0.3225, -0.245))
    steering_limit: float = math.radians(140)  # ±140°


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def swerve_ik(vx: float, vy: float, omega: float, wheel: WheelPosition,
              limit: float) -> tuple[float, float]:
    """Inverse kinematics for a single swerve wheel.

    Returns (speed_m_s, steer_degrees).
    """
    vix = vx - omega * wheel.y
    viy = vy + omega * wheel.x
    speed = math.sqrt(vix ** 2 + viy ** 2)

    if speed < 1e-6:
        return 0.0, 0.0

    theta = math.atan2(viy, vix)

    if theta < -limit or theta > limit:
        theta = math.atan2(-viy, -vix)
        speed = -speed

    return speed, math.degrees(theta)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SwervePoseController(Node):
    def __init__(self):
        super().__init__("swerve_pose_controller")

        # ----- ROS2 parameters -----
        self.declare_parameters(
            namespace="",
            parameters=[
                ("pid_x_kp", 1.0),
                ("pid_x_ki", 0.0),
                ("pid_x_kd", 0.1),
                ("pid_y_kp", 1.0),
                ("pid_y_ki", 0.0),
                ("pid_y_kd", 0.1),
                ("pid_theta_kp", 1.5),
                ("pid_theta_ki", 0.0),
                ("pid_theta_kd", 0.1),
                ("max_linear_vel", 0.3),
                ("max_angular_vel", 0.5),
                ("position_tolerance", 0.02),
                ("angle_tolerance", 0.03),
                ("control_rate", 20.0),
                ("odom_timeout", 1.0),
                ("goal_timeout", 30.0),
            ],
        )

        # ----- PID controllers -----
        self.pid_x = PIDState(
            kp=self.get_parameter("pid_x_kp").value,
            ki=self.get_parameter("pid_x_ki").value,
            kd=self.get_parameter("pid_x_kd").value,
        )
        self.pid_y = PIDState(
            kp=self.get_parameter("pid_y_kp").value,
            ki=self.get_parameter("pid_y_ki").value,
            kd=self.get_parameter("pid_y_kd").value,
        )
        self.pid_theta = PIDState(
            kp=self.get_parameter("pid_theta_kp").value,
            ki=self.get_parameter("pid_theta_ki").value,
            kd=self.get_parameter("pid_theta_kd").value,
        )

        # ----- Limits -----
        self.max_linear_vel = self.get_parameter("max_linear_vel").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.position_tolerance = self.get_parameter("position_tolerance").value
        self.angle_tolerance = self.get_parameter("angle_tolerance").value
        self.odom_timeout_sec = self.get_parameter("odom_timeout").value
        self.goal_timeout_sec = self.get_parameter("goal_timeout").value

        # ----- Swerve geometry -----
        self.geometry = SwerveGeometry()

        # ----- State -----
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.last_odom_time = None

        self.target_pose = None  # Pose2D or None
        self.target_received_time = None

        # ----- Subscribers -----
        self.create_subscription(Pose2D, "/pose_target", self._target_cb, 10)
        self.create_subscription(Odometry, "/odom_sim", self._odom_cb, 10)

        # ----- Publisher -----
        self.wheel_pub = self.create_publisher(
            Float32MultiArray, "/amr/wheel_pose_control", 10
        )

        # ----- Control timer -----
        rate_hz = self.get_parameter("control_rate").value
        self.dt = 1.0 / rate_hz
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f"SwervePoseController started — control rate {rate_hz} Hz"
        )

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.last_odom_time = self.get_clock().now()

    def _target_cb(self, msg: Pose2D):
        self.target_pose = msg
        self.target_received_time = self.get_clock().now()
        # Reset integrals to avoid accumulated error spikes
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_theta.reset()
        self.get_logger().info(
            f"New target: x={msg.x:.3f}, y={msg.y:.3f}, theta={msg.theta:.3f}"
        )

    # -----------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------

    def _control_loop(self):
        now = self.get_clock().now()

        # No target yet
        if self.target_pose is None:
            return

        # Odom timeout
        if self.last_odom_time is None or (
            (now - self.last_odom_time).nanoseconds / 1e9 > self.odom_timeout_sec
        ):
            self._publish_zero()
            return

        # Goal timeout
        if self.target_received_time is not None and (
            (now - self.target_received_time).nanoseconds / 1e9 > self.goal_timeout_sec
        ):
            self.get_logger().warn("Goal timeout — stopping", throttle_duration_sec=5.0)
            self._publish_zero()
            return

        # ----- Error in world frame -----
        ex_world = self.target_pose.x - self.current_x
        ey_world = self.target_pose.y - self.current_y
        etheta = normalize_angle(self.target_pose.theta - self.current_theta)

        pos_error = math.sqrt(ex_world ** 2 + ey_world ** 2)

        # Goal reached
        if pos_error < self.position_tolerance and abs(etheta) < self.angle_tolerance:
            self._publish_zero()
            return

        # ----- Rotate error into body frame -----
        cos_t = math.cos(-self.current_theta)
        sin_t = math.sin(-self.current_theta)
        ex_body = cos_t * ex_world - sin_t * ey_world
        ey_body = sin_t * ex_world + cos_t * ey_world

        # ----- PID -----
        vx_body = self.pid_x.compute(ex_body, self.dt)
        vy_body = self.pid_y.compute(ey_body, self.dt)
        omega = self.pid_theta.compute(etheta, self.dt)

        # ----- Clamp velocities -----
        linear_speed = math.sqrt(vx_body ** 2 + vy_body ** 2)
        if linear_speed > self.max_linear_vel:
            scale = self.max_linear_vel / linear_speed
            vx_body *= scale
            vy_body *= scale

        omega = max(-self.max_angular_vel, min(self.max_angular_vel, omega))

        # ----- Swerve IK per wheel -----
        limit = self.geometry.steering_limit
        fw_speed, fw_steer = swerve_ik(
            vx_body, vy_body, omega, self.geometry.front, limit
        )
        rl_speed, rl_steer = swerve_ik(
            vx_body, vy_body, omega, self.geometry.rear_left, limit
        )
        rr_speed, rr_steer = swerve_ik(
            vx_body, vy_body, omega, self.geometry.rear_right, limit
        )

        # ----- Publish -----
        msg = Float32MultiArray()
        msg.data = [
            fw_speed, fw_steer,
            rl_speed, rl_steer,
            rr_speed, rr_steer,
        ]
        self.wheel_pub.publish(msg)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _publish_zero(self):
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.wheel_pub.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SwervePoseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
