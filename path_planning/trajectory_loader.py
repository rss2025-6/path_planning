#!/usr/bin/env python3
import rclpy
import time
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import math

from path_planning.utils import LineTrajectory


class LoadTrajectory(Node):
    """ Loads a trajectory from the file system and publishes it to a ROS topic.
    """

    def __init__(self):
        super().__init__("trajectory_loader")

        self.declare_parameter("trajectory", "default")
        self.path = self.get_parameter("trajectory").get_parameter_value().string_value

        # initialize and load the trajectory
        self.trajectory = LineTrajectory(self, "/loaded_trajectory")
        self.get_logger().info(f"Loading from {self.path}")
        self.trajectory.load(self.path)

        self.pub_topic = "/trajectory/current"
        self.traj_pub = self.create_publisher(PoseArray, self.pub_topic, 1)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # visualize the loaded trajectory
        self.trajectory.publish_viz()

        # send the trajectory
        self.publish_trajectory()

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
            If True, return the distance, else returns None with False.
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


    def publish_trajectory(self):
        print("Publishing trajectory to:", self.pub_topic)
        self.traj_pub.publish(self.trajectory.toPoseArray())


def main(args=None):
    rclpy.init(args=args)
    load_trajectory = LoadTrajectory()
    rclpy.spin(load_trajectory)
    load_trajectory.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
