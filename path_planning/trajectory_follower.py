import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
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

        self.lookahead = 0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #

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
        self.command_forward_speed = 1.0 # desired speed, in m/s
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
    
    def circle_distance(self, circ_center, circ_radius, point1, point2):
        '''
        From: https://codereview.stackexchange.com/a/86428

        args:
            circ_center: size 2 numpy-vector of the form [x,y]^T
            circ_radius: scalar value
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
        c = (point1*point1) + (circ_center*circ_center) - (2*point1*circ_center)-(circ_radius**2)

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
    
    def pure_pursuit_controller(self, current_speed, L, angle_wrt_desired, lfw, Lfw):
        '''
        Only works in the forward motion.

        args:
            current_speed: The velocity that the racecar is currently travelling at, in m/s.
            L: the length between the rear and front axles, in meters.
            angle_wrt_desired: angle between direction robot is facing and Lfw, in radians.
            lfw: The distqance in front of the rear axle where, setting lfw = 0
                results in the conventional pure-pursuit controller.
            Lfw: The lookahead distance from lfw to desired point on path.
        returns:
            command_steering_angle: steering angle needed to hit point on desired path.
            command_speed: Speed given using PI controller.
            
        '''
        # Compute angle
        command_steering_angle = -math.atan((L*math.sin(angle_wrt_desired))/(Lfw/2+lfw*math.cos(angle_wrt_desired)))
        # Clip to acceptable angle range
        command_steering_angle = max(self.min_steering_angle, min(self.max_steering_angle, command_steering_angle))

        # PI controller for speed
        current_time = self.get_clock().now().nanoseconds
        delta_time = (current_time - self.speed_last_time) * 1e-9
        self.speed_last_time = current_time
        
        speed_error = self.command_forward_speed - current_speed
        self.speed_integral += delta_time * speed_error
        command_speed = self.speed_Kp * speed_error + self.speed_Ki * self.speed_integral
        
        
        return command_steering_angle, command_speed

    def pose_callback(self, odometry_msg):
        raise NotImplementedError

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
