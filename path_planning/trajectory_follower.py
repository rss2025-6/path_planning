import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray, Pose

import numpy as np
import math

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1.0  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = .3  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        # Extra parameters for controller
        self.min_steering_angle = -math.pi/3 # Need to change to real angle
        self.max_steering_angle = math.pi/3 # Need to change to real angle

        self.speed_Kp = 0.0 # update
        self.speed_Ki = 0.0 # update
        self.speed_integral = 0.0
        self.speed_last_time = self.get_clock().now().nanoseconds


    def distance(self, x1, y1, x2, y2, x3, y3):
        '''
        From: https://stackoverflow.com/a/2233538

        Returns the shortest distance from a point to a line, 
        where the line is defined by 2 points (x1, y1) and (x2, y2)
        and the point is (x3,y3).
        '''
        px = x2-x1
        py = y2-y1

        norm = px**2 + py**2

        u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        # Note: If the actual distance does not matter,
        # if you only want to compare what this function
        # returns to other results of this function, you
        # can just return the squared distance instead
        # (i.e. remove the sqrt) to gain a little performance

        dist = (dx*dx + dy*dy)**.5

        return dist
    
    def circle_distance(self, circ_center, point1, point2):
        '''
        From: https://codereview.stackexchange.com/a/86428

        args:
            circ_center: size 2 numpy-vector of the form [x,y]^T
            point1: Start of line segment
                    size 2 numpy-vector of the form [x,y]^T
            point2: End of line segment
                    size 2 numpy-vector of the form [x,y]^T
        returns:
            True if the circle collides with the line segment, False otherwise
            If True, return the point of intersection, else returns None with False.
        '''
        V = point2-point1
        a=V*V
        b = 2*V*(point1-circ_center)
        c = (point1*point1) + (circ_center*circ_center) - (2*point1*circ_center)-(self.lookahead**2)

        discriminant = b**2 - 4* a * c
        if discriminant < 0:
            # Line is missing circle entirely
            return False, None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
            # The line segment misses the circle (but would miss it if extended)
            return False, None
        
        t = max(0, min(1, - b / (2 * a)))
        return True, point1 + t * V
    
    def pure_pursuit_controller(self, current_speed, angle_wrt_desired, lfw):
        '''
        Only works in the forward motion.

        args:
            current_speed: The velocity that the racecar is currently travelling at, in m/s.
            angle_wrt_desired: angle between direction robot is facing and Lfw, in radians.
            lfw: The distance in front of the rear axle where, setting lfw = 0
                results in the conventional pure-pursuit controller.
        returns:
            command_steering_angle: steering angle needed to hit point on desired path.
            command_speed: Speed given using PI controller.
            
        '''
        # Compute angle
        command_steering_angle = -math.atan((self.wheelbase_length*math.sin(angle_wrt_desired))/(self.lookahead/2+lfw*math.cos(angle_wrt_desired)))
        # Clip to acceptable angle range
        command_steering_angle = max(self.min_steering_angle, min(self.max_steering_angle, command_steering_angle))

        # PI controller for speed
        current_time = self.get_clock().now().nanoseconds
        delta_time = (current_time - self.speed_last_time) * 1e-9
        self.speed_last_time = current_time
        
        speed_error = self.speed - current_speed
        self.speed_integral += delta_time * speed_error
        command_speed = self.speed_Kp * speed_error + self.speed_Ki * self.speed_integral
        
        
        return command_steering_angle, command_speed

    def pose_callback(self, odometry_msg):
        robot_pose = odometry_msg.pose.pose.position
        robot_orient = odometry_msg.pose.pose.orientation
        velocity = odometry_msg.twist.twist.linear
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2)

        robot_pose_arr = np.array([robot_pose.x, robot_pose.y])

        _, _, yaw = euler_from_quaternion([robot_orient.x, robot_orient.y, self.robot_orient.z, self.robot_orient.w])

        if self.initialized_traj:
            points = self.points
            
            distances = np.zeros(len(points)-1)
            for i in range(len(points)-1):
                v = points[i]
                w = points[i + 1]
                distances[i] = self.distance(v[0],v[1],w[0],w[1],robot_pose_arr[0],self.robot_pose_arr[1])
            
            min_ind = np.argmin(distances)

            for i in range(min_ind, len(points)-1):
                p1 = points[i]
                p2 = points[i + 1]
                found, intersect_pt = self.circle_distance(robot_pose_arr, p1, p2)
                if found:
                    self.draw_marker(intersect_pt[0], intersect_pt[1])

                    yaw_2 = np.atan(intersect_pt[1], intersect_pt[0])
                    phi = np.acos(np.cos(yaw)*np.cos(yaw_2)-np.sin(yaw)*np.sin(yaw_2))

                    steer, speed = self.pure_pursuit_controller(current_speed, phi, 0)
                    # create drive command
                    driveCommand = AckermannDriveStamped()
                    driveCommand.header.frame_id = "base_link"
                    driveCommand.header.stamp = self.get_clock().now().to_msg()
                    driveCommand.drive.steering_angle= steer
                    driveCommand.drive.speed= speed

                    # publish drive command
                    self.drive_pub.publish(driveCommand)

                    self.get_logger().info(f"Steering Angle = {steer*180/np.pi} deg")
                    self.get_logger().info(f"Speed = {speed} m/s")
                    return
            
            self.get_logger().info("NOT FOUND")

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

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


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
