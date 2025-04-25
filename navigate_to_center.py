import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import cv2.aruco as aruco
import time

class NavigateToCenter(Node):
    def __init__(self):
        super().__init__('navigate_to_center')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, qos_profile)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.control_loop)

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        self.marker_positions = {}  # Global positions of markers
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # Assume robot starts at origin
        self.center_point = None
        self.is_rotating = True
        self.rotation_start_time = self.get_clock().now()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                aruco.drawDetectedMarkers(cv_image, corners, ids)
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    center_x = (corner[0][0][0] + corner[0][2][0]) / 2
                    center_y = (corner[0][0][1] + corner[0][2][1]) / 2
                    marker_camera_pos = np.array([center_x, center_y, 0.0])

                    if marker_id not in self.marker_positions:
                        if len(self.marker_positions) == 0:
                            self.marker_positions[marker_id] = np.array([0.0, 0.0, 0.0])  # Origin
                            self.get_logger().info(f"Marker {marker_id} set as origin.")
                        else:
                            global_position = self.calculate_global_position(marker_camera_pos)
                            self.marker_positions[marker_id] = global_position
                            self.get_logger().info(f"Marker {marker_id} global position: {global_position}")

            cv2.imshow("Camera Image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def calculate_global_position(self, marker_camera_pos):
        robot_x, robot_y, robot_theta = self.robot_pose
        rotation_matrix = np.array([
            [np.cos(robot_theta), -np.sin(robot_theta)],
            [np.sin(robot_theta), np.cos(robot_theta)]
        ])
        global_position = np.dot(rotation_matrix, marker_camera_pos[:2]) + np.array([robot_x, robot_y])
        return np.array([global_position[0], global_position[1], 0.0])

    def control_loop(self):
        if self.is_rotating:
            cmd = Twist()
            cmd.angular.z = 0.3
            self.cmd_vel_pub.publish(cmd)

            elapsed_time = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9
            if elapsed_time >= 2 * np.pi / 0.3:
                self.is_rotating = False
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info("Finished rotating 360 degrees.")
                if len(self.marker_positions) < 4:
                    self.get_logger().info("Not all markers found. Moving to the midpoint of detected markers.")
                    self.move_to_midpoint()
            return

        if self.center_point is None and len(self.marker_positions) == 4:
            self.center_point = self.calculate_center()
            self.get_logger().info(f"Center point calculated: {self.center_point}")

        if self.center_point is not None:
            self.navigate_to_center()

    def move_to_midpoint(self):
        if len(self.marker_positions) > 0:
            positions = list(self.marker_positions.values())
            midpoint = np.mean(positions, axis=0)
            self.get_logger().info(f"Moving to midpoint at position {midpoint}.")
            self.navigate_to_position(midpoint)
            self.is_rotating = True
            self.rotation_start_time = self.get_clock().now()

    def navigate_to_position(self, position):
        self.get_logger().info(f"Navigating to position {position}.")
        error_x = position[0] - self.robot_pose[0]
        error_y = position[1] - self.robot_pose[1]
        distance = np.sqrt(error_x**2 + error_y**2)

        target_angle = np.arctan2(error_y, error_x)
        angle_error = target_angle - self.robot_pose[2]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        cmd = Twist()
        start_time = time.time()

        while abs(angle_error) > 0.1:
            if time.time() - start_time > 10:  # 超时 10 秒退出
                self.get_logger().warn("Timeout while rotating to target angle.")
                break
            cmd.angular.z = 0.3 * angle_error
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f"Publishing cmd_vel: angular.z={cmd.angular.z}")
            angle_error = target_angle - self.robot_pose[2]
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info("Finished rotating to target angle.")

        start_time = time.time()
        while distance > 1.0:
            if time.time() - start_time > 20:  # 超时 20 秒退出
                self.get_logger().warn("Timeout while moving to target position.")
                break
            self.get_logger().info(f"Moving towards target. Distance: {distance}")
            cmd.linear.x = 0.2
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f"Publishing cmd_vel: linear.x={cmd.linear.x}")
            error_x = position[0] - self.robot_pose[0]
            error_y = position[1] - self.robot_pose[1]
            distance = np.sqrt(error_x**2 + error_y**2)

        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info("Reached the target position.")

    def calculate_center(self):
        positions = list(self.marker_positions.values())
        x = sum(p[0] for p in positions) / len(positions)
        y = sum(p[1] for p in positions) / len(positions)
        return np.array([x, y])

    def navigate_to_center(self):
        error_x = self.center_point[0] - self.robot_pose[0]
        error_y = self.center_point[1] - self.robot_pose[1]
        distance = np.sqrt(error_x**2 + error_y**2)

        if distance < 0.1:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info("Reached the center point!")
            return

        cmd = Twist()
        cmd.linear.x = 0.2 * error_x / distance
        cmd.angular.z = 0.5 * np.arctan2(error_y, error_x)
        self.get_logger().info(f"Publishing cmd_vel: linear.x={cmd.linear.x}, angular.z={cmd.angular.z}")
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavigateToCenter()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()