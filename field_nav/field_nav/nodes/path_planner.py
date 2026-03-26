"""
Row Following Path Planner
--------------------------
Stanley-method lateral controller for following a crop row.

The Stanley controller (used by Stanford's DARPA car) computes steering
as a function of both heading error and cross-track error:

    δ = ψ_e + arctan(k · e / v)

where
    ψ_e  = heading error relative to row  [rad]
    e    = lateral cross-track error       [m]
    v    = forward speed                   [m/s]
    k    = gain                            (tunable)

This approach is well-suited to slow agricultural robots because it
remains stable at near-zero speeds with a softened denominator.

Subscribes
----------
/crop_row/deviation   std_msgs/Float32    - lateral error from row centre
/crop_row/heading     std_msgs/Float32    - heading error
/ekf/odom             nav_msgs/Odometry   - current speed for Stanley term

Publishes
---------
/cmd_vel              geometry_msgs/Twist - velocity commands to base
/planner/status       std_msgs/String     - human-readable state
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class RowFollowingPlanner(Node):
    """
    Stanley lateral controller that drives the robot along a detected crop row.

    States
    ------
    IDLE         → waiting for a row detection
    FOLLOWING    → actively tracking the row
    END_OF_ROW   → no line detected for > timeout_s seconds
    """

    def __init__(self):
        super().__init__("row_following_planner")

        # --- params ---
        self.declare_parameter("target_speed_mps",   0.5)
        self.declare_parameter("stanley_gain_k",     1.2)
        self.declare_parameter("max_steer_rad",      0.52)    # ~30°
        self.declare_parameter("speed_softener",     0.3)     # prevents div/0 at low v
        self.declare_parameter("row_lost_timeout_s", 1.5)

        self._v_target   = self.get_parameter("target_speed_mps").value
        self._k          = self.get_parameter("stanley_gain_k").value
        self._max_steer  = self.get_parameter("max_steer_rad").value
        self._softener   = self.get_parameter("speed_softener").value
        self._lost_to    = self.get_parameter("row_lost_timeout_s").value

        self._cross_track_err = 0.0
        self._heading_err     = 0.0
        self._current_speed   = 0.0
        self._last_row_time   = None
        self._state           = "IDLE"

        # --- subscriptions ---
        self.create_subscription(Float32, "/crop_row/deviation", self._dev_cb,  10)
        self.create_subscription(Float32, "/crop_row/heading",   self._hdg_cb,  10)
        self.create_subscription(Odometry, "/ekf/odom",          self._odom_cb, 10)

        # --- publishers ---
        self._cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",         10)
        self._status_pub = self.create_publisher(String, "/planner/status",  10)

        self._control_timer = self.create_timer(0.05, self._control_loop)   # 20 Hz

        self.get_logger().info("Row Following Planner online ✓")

    # ------------------------------------------------------------------
    def _dev_cb(self, msg: Float32):
        self._cross_track_err = msg.data
        self._last_row_time   = self.get_clock().now().nanoseconds * 1e-9

    def _hdg_cb(self, msg: Float32):
        self._heading_err = msg.data

    def _odom_cb(self, msg: Odometry):
        self._current_speed = msg.twist.twist.linear.x

    # ------------------------------------------------------------------
    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        cmd = Twist()

        # --- state machine ---
        row_detected = (
            self._last_row_time is not None
            and (now - self._last_row_time) < self._lost_to
        )

        if not row_detected:
            self._state = "END_OF_ROW" if self._last_row_time else "IDLE"
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self._cmd_pub.publish(cmd)
            self._status_pub.publish(String(data=self._state))
            return

        self._state = "FOLLOWING"

        # --- Stanley controller ---
        e   = self._cross_track_err
        psi = self._heading_err
        v   = max(abs(self._current_speed), self._softener)

        stanley_angle = psi + math.atan2(self._k * e, v)
        steer = max(-self._max_steer, min(self._max_steer, stanley_angle))

        cmd.linear.x  = self._v_target
        cmd.angular.z = steer        # differential drive: ω ≈ steer at low speed

        self._cmd_pub.publish(cmd)
        self._status_pub.publish(String(data=f"FOLLOWING | e={e:.3f}m ψ={math.degrees(psi):.1f}°"))

        if abs(e) > 0.3 or abs(psi) > 0.4:
            self.get_logger().warn(
                f"Large deviation: cross_track={e:.3f}m  heading_err={math.degrees(psi):.1f}°"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RowFollowingPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
