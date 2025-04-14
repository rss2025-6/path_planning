import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64

import numpy as np
import math

from .utils import LineTrajectory

sin = np.sin
cos = np.cos

atan2 = np.arctan2
acos = np.arccos

pi = np.pi

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 2.0  # FILL IN # RADIUS OF CIRCLE
        self.speed = 1.  # FILL IN #
        self.wheelbase_length = 0.33  # FILL IN #
        self.rear_axle_offset = self.wheelbase_length / 2 # when self.rear_axle_offset = 0.0 then we have standard pure pursuit

        self.steer = 0.0
        self.steer_max = pi/4

        # For calculating lookahead
        self.min_lookahead = self.wheelbase_length/np.tan(self.steer_max)
        self.max_lookahead = 4 * self.speed
        
        # Distance from the end goal at which the robot is done following the path (i.e. stops moving)
        self.goal_final_distance_to_end = self.wheelbase_length/2

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.error_pub = self.create_publisher(Float64, "/error", 1)
        
        self.initialized_traj = False

        self.marker_pub = self.create_publisher(Marker, "/point_marker", 1)
        self.traj_points_pub = self.create_publisher(PoseArray, "/traj_poses", 1)
    
    def calculate_lookahead(self, distance_to_end):
        R = self.wheelbase_length/np.tan(self.steer)
        return max(self.min_lookahead, min(R, self.max_lookahead, distance_to_end))

    # return distance between 2 points
    def distance(self, p, v):
        return np.sqrt(np.sum((p - v)**2))
    
    def eta(self, v1, v2):
        # self.get_logger().info(f"v1 = {v1}, v2 = {v2}")
        return acos(v1.T @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def draw_marker(self, x, y):
        """
        Publish a marker to represent the cone in rviz
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        self.marker_pub.publish(marker)
    
    # Returns the minimum distance b/t line segment vw and point p
    # where v = np.array(x1,y1), w = np.array(x2,y2)
    def min_dist(self, v, w):
        p = self.robot_pose_arr

        l2 = np.sum((v - w)**2)

        if (l2 == 0.0):
            return self.distance(p, v)
        
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)

        return self.distance(p, projection)
    
    def find_lookahead_intersect(self, p1, p2):
        Q = self.robot_pose_arr                 # Centre of circle = robot pose
        r = self.lookahead                  # Radius of circle = radius of robot
        V = p2 - p1  # Vector along line segment 

        a = V.T@V
        b = 2 * V.T@(p1 - Q)
        c = p1.T@p1 + Q.T@Q - 2 * p1.T@Q - r**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            return False, None
        
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
            return False, None
        
        t = max(0, min(1, - b / (2 * a)))

        return True, p1 + t * V
            
    def pose_callback(self, odometry_msg):

        self.robot_pose = odometry_msg.pose.pose.position
        self.robot_orient = odometry_msg.pose.pose.orientation

        self.robot_pose_arr = np.array([self.robot_pose.x, self.robot_pose.y])

        r, p, self.yaw = euler_from_quaternion([self.robot_orient.x, self.robot_orient.y, self.robot_orient.z, self.robot_orient.w])

        if self.initialized_traj:
            points = self.points

            distances = np.zeros(len(points) - 1) # so dist[i] = distance b/t traj pt i and pt i + 1
            for i in range(len(points) - 1):
                v = points[i]
                w = points[i + 1]
                distances[i] = self.min_dist(v, w)
            
            min_ind = np.argmin(distances)

            distance_to_end = distances[len(distances)-1]

            # If within self.goal_final_distance_to_end of the end, then stop moving
            if (distance_to_end < self.goal_final_distance_to_end):
                # create drive command
                driveCommand = AckermannDriveStamped()
                driveCommand.header.frame_id = "base_link"
                driveCommand.header.stamp = self.get_clock().now().to_msg()
                driveCommand.drive.steering_angle = 0.0
                driveCommand.drive.speed = 0.0

                # publish drive command
                self.drive_pub.publish(driveCommand)
                return

            # Publish error
            error_msg = Float64()
            error_msg.data = distances[min_ind]
            self.error_pub.publish(error_msg)

            for i in range(min_ind, len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                found, intersect_pt = self.find_lookahead_intersect(p1, p2)
                if found:
                    break

            if found:

                self.draw_marker(intersect_pt[0], intersect_pt[1])

                yaw_2 = atan2(intersect_pt[1] - self.robot_pose_arr[1], intersect_pt[0] - self.robot_pose_arr[0])
                eta = yaw_2 - self.yaw 
                
                # Standard pure pursuit (w/o self.rear_axle_offset)
                # steer = atan2(2 * self.wheelbase_length * sin(eta), self.lookahead)
                # Optimized pure pursuit
                self.steer = atan2(self.wheelbase_length * sin(eta), (self.lookahead/2)+(self.rear_axle_offset*cos(eta)))

                # Update the lookahead
                self.lookahead = self.calculate_lookahead(distance_to_end)

                # create drive command
                driveCommand = AckermannDriveStamped()
                driveCommand.header.frame_id = "base_link"
                driveCommand.header.stamp = self.get_clock().now().to_msg()
                driveCommand.drive.steering_angle= self.steer
                driveCommand.drive.speed= self.speed
                
                # publish drive command
                self.drive_pub.publish(driveCommand)
                
                self.get_logger().info(f"steer = {self.steer}")
            else:
                self.get_logger().info("NOT FOUND")

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.points = np.array(self.trajectory.points)

        self.initialized_traj = True
        


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
