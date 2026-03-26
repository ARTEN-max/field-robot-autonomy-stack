"""
Crop Row Detector Node
----------------------
Processes front-facing camera frames to estimate lateral deviation
and heading error relative to the current crop row.

Pipeline
--------
1. Undistort & crop to ROI (lower half of image — ground view)
2. Convert to HSV → green-channel mask  (healthy canopy)
3. Canny edge detection + Hough line transform
4. Fit dominant vanishing-point lines → estimate row centreline
5. Publish deviation [m] and heading error [rad]

Subscribes
----------
/camera/image_raw   sensor_msgs/Image       - raw RGB frame
/camera/info        sensor_msgs/CameraInfo  - intrinsics

Publishes
---------
/crop_row/deviation std_msgs/Float32        - lateral offset from row centre [m]
/crop_row/heading   std_msgs/Float32        - heading error [rad]
/crop_row/debug     sensor_msgs/Image       - annotated frame (optional)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class CropRowDetector(Node):
    """
    Detects crop rows via green-band Hough lines and
    publishes lateral / heading deviation for the path controller.
    """

    def __init__(self):
        super().__init__("crop_row_detector")

        # --- tunable params ---
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("roi_top_fraction", 0.5)   # ignore sky
        self.declare_parameter("hough_threshold", 60)
        self.declare_parameter("min_line_length", 80)
        self.declare_parameter("max_line_gap", 25)
        self.declare_parameter("camera_height_m", 0.55)   # lens above ground

        self._debug   = self.get_parameter("publish_debug_image").value
        self._roi_frac = self.get_parameter("roi_top_fraction").value
        self._cam_h   = self.get_parameter("camera_height_m").value

        self._bridge = CvBridge()
        self._fx = None   # filled once CameraInfo arrives

        # --- subscriptions ---
        self.create_subscription(Image,      "/camera/image_raw", self._image_cb, 2)
        self.create_subscription(CameraInfo, "/camera/info",      self._info_cb,  1)

        # --- publishers ---
        self._dev_pub  = self.create_publisher(Float32, "/crop_row/deviation", 10)
        self._hdg_pub  = self.create_publisher(Float32, "/crop_row/heading",   10)
        self._dbg_pub  = self.create_publisher(Image,   "/crop_row/debug",     2)

        self.get_logger().info("Crop Row Detector online ✓")

    # ------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo):
        """Cache focal length once."""
        if self._fx is None:
            self._fx = msg.k[0]   # K[0,0]
            self.get_logger().info(f"Camera fx={self._fx:.1f}px")

    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]

        # 1. ROI — lower portion of frame only
        roi_y = int(h * self._roi_frac)
        roi   = frame[roi_y:, :]

        # 2. Green-channel vegetation mask (ExG index)
        b, g, r = cv2.split(roi.astype(np.float32))
        exg  = 2 * g - r - b
        mask = np.clip(exg, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

        # 3. Canny edges on masked region
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 80)

        # 4. Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.get_parameter("hough_threshold").value,
            minLineLength=self.get_parameter("min_line_length").value,
            maxLineGap=self.get_parameter("max_line_gap").value,
        )

        deviation_m = 0.0
        heading_err = 0.0

        if lines is not None:
            left_lines, right_lines = self._classify_lines(lines, w)
            cx_img = w / 2.0

            left_x  = self._average_x_at_bottom(left_lines,  roi.shape[0]) if left_lines  else None
            right_x = self._average_x_at_bottom(right_lines, roi.shape[0]) if right_lines else None

            if left_x is not None and right_x is not None:
                row_centre = (left_x + right_x) / 2.0
            elif left_x is not None:
                row_centre = left_x + w * 0.25
            elif right_x is not None:
                row_centre = right_x - w * 0.25
            else:
                row_centre = cx_img

            px_offset = row_centre - cx_img
            if self._fx:
                deviation_m = (px_offset / self._fx) * self._cam_h
            else:
                deviation_m = px_offset / (w / 2.0)   # normalised fallback

            heading_err = self._estimate_heading(left_lines, right_lines, w)

            if self._debug:
                self._draw_debug(roi, lines, left_x, right_x, row_centre, roi_y, frame)

        self._dev_pub.publish(Float32(data=float(deviation_m)))
        self._hdg_pub.publish(Float32(data=float(heading_err)))

        if self._debug:
            self._dbg_pub.publish(self._bridge.cv2_to_imgmsg(frame, "bgr8"))

    # ------------------------------------------------------------------
    @staticmethod
    def _classify_lines(lines, img_width):
        """Split Hough lines into left / right sets by slope."""
        left, right = [], []
        cx = img_width / 2
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:     # near-horizontal — skip
                continue
            mid_x = (x1 + x2) / 2
            if mid_x < cx:
                left.append(line[0])
            else:
                right.append(line[0])
        return left, right

    @staticmethod
    def _average_x_at_bottom(lines, img_h):
        """Extrapolate each line to the bottom of ROI, return mean x."""
        xs = []
        for x1, y1, x2, y2 in lines:
            if y2 == y1:
                continue
            # x at y = img_h
            x_bot = x1 + (x2 - x1) * (img_h - y1) / (y2 - y1)
            xs.append(x_bot)
        return float(np.mean(xs)) if xs else None

    @staticmethod
    def _estimate_heading(left_lines, right_lines, img_w):
        """Average heading angle of detected row lines [rad]."""
        angles = []
        for line in (left_lines or []) + (right_lines or []):
            x1, y1, x2, y2 = line
            angles.append(np.arctan2(y2 - y1, x2 - x1))
        if not angles:
            return 0.0
        mean_angle = float(np.mean(angles))
        # Rows are roughly vertical → deviation from π/2
        return mean_angle - (np.pi / 2)

    def _draw_debug(self, roi, lines, left_x, right_x, centre_x, roi_y, full_frame):
        offset = roi_y
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(full_frame, (x1, y1 + offset), (x2, y2 + offset), (0, 255, 0), 1)
        h_full = full_frame.shape[0]
        if left_x is not None:
            cv2.line(full_frame, (int(left_x), h_full - 10), (int(left_x), h_full - 30), (255, 0, 0), 2)
        if right_x is not None:
            cv2.line(full_frame, (int(right_x), h_full - 10), (int(right_x), h_full - 30), (0, 0, 255), 2)
        cx = int(centre_x)
        cv2.line(full_frame, (cx, h_full - 10), (cx, h_full - 40), (0, 255, 255), 2)


def main(args=None):
    rclpy.init(args=args)
    node = CropRowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
