"""
EKF Localizer Node
------------------
Fuses wheel odometry and GPS into a smooth pose estimate
for a differential-drive agricultural robot navigating crop rows.

State vector:  x = [x, y, θ]ᵀ   (field-frame)
Motion model:  dead-reckoning via encoder ticks (non-linear)
Measurements:  GPS position (x, y) at ~10 Hz
"""

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64MultiArray

from field_nav.utils.math_utils import wrap_angle, latlon_to_local_xy


class EKFLocalizer(Node):
    """
    Extended Kalman Filter fusing odometry + GPS for field navigation.

    Subscribes
    ----------
    /odom            nav_msgs/Odometry          - wheel encoder odometry
    /gps/fix         sensor_msgs/NavSatFix       - raw GPS fix

    Publishes
    ---------
    /ekf/pose        geometry_msgs/PoseWithCovarianceStamped
    /ekf/odom        nav_msgs/Odometry
    """

    # --- noise tuning knobs (override via params.yaml) ---
    _Q_DIAG   = (0.05, 0.05, 0.01)   # process noise:  x[m], y[m], θ[rad]
    _R_GPS    = (2.0,  2.0)           # GPS meas noise: x[m], y[m]

    def __init__(self):
        super().__init__("ekf_localizer")

        # --- declare & load params ---
        self.declare_parameter("gps_origin_lat", 43.4643)   # default: Waterloo, ON
        self.declare_parameter("gps_origin_lon", -80.5204)
        self.declare_parameter("publish_rate_hz", 50.0)

        self._origin_lat = self.get_parameter("gps_origin_lat").value
        self._origin_lon = self.get_parameter("gps_origin_lon").value

        # --- EKF state: [x, y, theta] ---
        self._x = np.zeros(3)
        self._P = np.diag([1.0, 1.0, 0.1])   # initial covariance

        self._Q = np.diag(self._Q_DIAG)
        self._R = np.diag(self._R_GPS)

        self._last_odom_time = None
        self._last_v = 0.0
        self._last_w = 0.0

        # --- ROS I/O ---
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self._gps_sub = self.create_subscription(
            NavSatFix, "/gps/fix", self._gps_callback, 10
        )
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ekf/pose", 10
        )
        self._odom_pub = self.create_publisher(Odometry, "/ekf/odom", 10)

        rate = self.get_parameter("publish_rate_hz").value
        self._timer = self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info("EKF Localizer online ✓")

    # ------------------------------------------------------------------
    # Predict step  (called on every odometry message)
    # ------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_odom_time is None:
            self._last_odom_time = now
            return

        dt = now - self._last_odom_time
        self._last_odom_time = now

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        self._predict(v, w, dt)
        self._last_v, self._last_w = v, w

    def _predict(self, v: float, w: float, dt: float):
        x, y, th = self._x

        # --- state transition ---
        if abs(w) < 1e-6:                    # straight-line motion
            dx = v * dt * np.cos(th)
            dy = v * dt * np.sin(th)
            dth = 0.0
        else:                                 # arc motion
            r = v / w
            dth = w * dt
            dx = r * (np.sin(th + dth) - np.sin(th))
            dy = r * (-np.cos(th + dth) + np.cos(th))

        self._x = np.array([x + dx, y + dy, wrap_angle(th + dth)])

        # --- Jacobian of motion model wrt state ---
        if abs(w) < 1e-6:
            G = np.array([
                [1, 0, -v * dt * np.sin(th)],
                [0, 1,  v * dt * np.cos(th)],
                [0, 0,  1],
            ])
        else:
            r = v / w
            G = np.array([
                [1, 0, r * (np.cos(th + dth) - np.cos(th))],
                [0, 1, r * (np.sin(th + dth) - np.sin(th))],
                [0, 0, 1],
            ])

        self._P = G @ self._P @ G.T + self._Q

    # ------------------------------------------------------------------
    # Update step  (called on every GPS fix)
    # ------------------------------------------------------------------
    def _gps_callback(self, msg: NavSatFix):
        if msg.status.status < 0:            # no fix
            return

        meas_x, meas_y = latlon_to_local_xy(
            msg.latitude, msg.longitude,
            self._origin_lat, self._origin_lon,
        )

        z = np.array([meas_x, meas_y])
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])             # GPS only sees x, y

        y_res = z - H @ self._x              # innovation
        S = H @ self._P @ H.T + self._R      # innovation covariance
        K = self._P @ H.T @ np.linalg.inv(S) # Kalman gain

        self._x = self._x + K @ y_res
        self._x[2] = wrap_angle(self._x[2])
        self._P = (np.eye(3) - K @ H) @ self._P

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def _publish(self):
        now = self.get_clock().now().to_msg()
        x, y, th = self._x

        # PoseWithCovarianceStamped
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = float(x)
        pose_msg.pose.pose.position.y = float(y)
        pose_msg.pose.pose.orientation.z = float(np.sin(th / 2))
        pose_msg.pose.pose.orientation.w = float(np.cos(th / 2))

        # flatten 3x3 covariance into 6x6 (ROS convention)
        cov6 = np.zeros((6, 6))
        cov6[:2, :2] = self._P[:2, :2]
        cov6[:2, 5]  = self._P[:2, 2]
        cov6[5, :2]  = self._P[2, :2]
        cov6[5, 5]   = self._P[2, 2]
        pose_msg.pose.covariance = cov6.flatten().tolist()

        self._pose_pub.publish(pose_msg)

        # Odometry (for nav2 / rviz2)
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = float(self._last_v)
        odom_msg.twist.twist.angular.z = float(self._last_w)
        self._odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
