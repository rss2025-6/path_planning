import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

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

        self.lookahead = 0.4  # FILL IN # RADIUS OF CIRCLE
        self.speed = 1.  # FILL IN #
        self.wheelbase_length = 0.8  # FILL IN #

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
        
        self.initialized_traj = False

        self.steer_max = pi/4

    # return distance between 2 points
    def distance(self, p, v):
        return np.sqrt(np.sum((p - v)**2))
    
    def eta(self, v1, v2):
        self.get_logger().info(f"v1 = {v1}, v2 = {v2}")
        return acos(v1.T @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
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
    
    def find_lookahead(self, p1, p2):
        Q = self.robot_pose_arr                 # Centre of circle = robot pose
        r = self.lookahead                  # Radius of circle = radius of robot
        V = p2 - p1  # Vector along line segment 

        a = V.T@V # TODO: Check dims, want [,] * [[],[]]
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
            # line trajectory =
            # self.points: List[Tuple[float, float]] = []
            # self.distances = []
            # self.has_acceleration = False
            # self.visualize = False
            # self.viz_namespace = viz_namespace
            # self.node = node
            # points = self.trajectory.points # TODO: check types
            points = self.points
            
            # probably very inefficient
            distances = np.zeros((len(points), len(points)))
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        v = points[i]
                        w = points[j]
                        distances[i,j] = self.min_dist(v, w)
                    else:
                        distances[i,j] = 100000
            
            min_ind = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

            p1 = points[min_ind[0]]
            p2 = points[min_ind[1]]

            self.get_logger().info(f"CLOSEST PT = ({p1}, {p2})")

            found, intersect_pt = self.find_lookahead(p1, p2)

            if found:
                L1 = self.distance(self.robot_pose_arr, intersect_pt)

                vel_vect = np.array([self.speed * cos(self.yaw), self.speed * sin(self.yaw)])

                eta = self.eta(vel_vect, intersect_pt)
                steer = np.clip(atan2(L1, 2 * self.wheelbase_length * sin(eta)), -self.steer_max, self.steer_max)
                
                # create drive command
                driveCommand = AckermannDriveStamped()
                driveCommand.header.frame_id = "base_link"
                driveCommand.header.stamp = self.get_clock().now().to_msg()
                driveCommand.drive.steering_angle= steer
                driveCommand.drive.speed= self.speed
                
                # publish drive command
                self.drive_pub.publish(driveCommand)
                
                # self.get_logger().info(f"v = {steer}")
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
