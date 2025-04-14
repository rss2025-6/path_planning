import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion
import matplotlib.pyplot as plt

from .utils import LineTrajectory

import numpy as np
import heapq

sin = np.sin
cos = np.cos

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.goal_coords = None
        self.robot_pose = None
        self.robot_cell = None
        self.goal_cell = None

        self.cell_size = 20
        self.map_set = False

        self.edges = {}
        self.verticies = []

    # TODO: Don't use all the map points it's probably too much
    def map_cb(self, msg):    
        # Convert the map to a numpy array
        self.map = np.array(msg.data)
        
        self.map_resolution = msg.info.resolution

        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin

        x = self.map_origin.orientation.x
        y = self.map_origin.orientation.y
        z = self.map_origin.orientation.z
        w = self.map_origin.orientation.w
        r, p, self.map_yaw = euler_from_quaternion([x, y, z, w])

        self.map = np.reshape(self.map, (self.map_height, self.map_width))

        self.get_logger().info("MAP INITIALIZED")
        self.get_logger().info(f"SHAPE = {np.shape(self.map)}")
        self.get_logger().info(f"WIDTH = {self.map_width}")
        self.get_logger().info(f"WIDTH = {self.map_height}")

        self.map_set = True

    def xy_2_uv(self, pose):
        # uv = pose[:]
        # uv += np.array([self.map_origin.position.x, self.map_origin.position.y])
        # uv /= self.map_resolution

        # self.get_logger().info(f"POSE = {uv}")
        # self.get_logger().info(f"POSE ADJ = {uv - uv%self.cell_size}")
        uv = pose[:]
        uv -= np.array([self.map_origin.position.x, self.map_origin.position.y])
        R = np.array([[cos(self.map_yaw), -sin(self.map_yaw)], [sin(self.map_yaw), cos(self.map_yaw)]])
        uv = np.dot(R, uv)
        uv /= self.map_resolution

        self.get_logger().info(f"POSE = {uv}")
        self.get_logger().info(f"POSE ADJ = {uv - uv%self.cell_size}")

        return uv - uv%self.cell_size

    # TODO: HOW TO APPLY ROTATION
    def uv_2_xy(self, pose):
        xy = pose[:]
        xy *= self.map_resolution
        R = np.array([[cos(self.map_yaw), sin(self.map_yaw)], [-sin(self.map_yaw), cos(self.map_yaw)]])
        xy = np.dot(R, xy)
        xy += np.array([self.map_origin.position.x, self.map_origin.position.y])

        return xy

    # Record the robot pose
    def pose_cb(self, pose):
        self.robot_pose = np.array([pose.pose.pose.position.x, pose.pose.pose.position.y]) 

        self.robot_cell = self.xy_2_uv(self.robot_pose)

        self.get_logger().info("POSITION SET")

    # Record the goal coordinates
    def goal_cb(self, msg):
        self.goal_coords = np.array([msg.pose.position.x, msg.pose.position.y])

        self.goal_cell = self.xy_2_uv(self.goal_coords)

        self.get_logger().info("GOAL SET")

        self.plan_path(self.robot_cell, self.goal_cell, self.map)

    # Plot full path
    def plot_path(self, path):
        self.get_logger().info("PLOTTING PATH")
        fig, ax = plt.subplots()

        # Iterate through verticies
        for v in self.verticies:

            # Plot each vertex
            ax.scatter(v[0], v[1], color='black')

            # Get neighbors
            all_neigh = self.edges.get(tuple(v), [])

            # Plot edges between neighbors
            for neigh, _ in all_neigh:
                x1, y1 = v
                x2, y2 = neigh
                ax.plot([x1, x2], [y1, y2], color='blue')

        # Plot the path
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=5)

        ax.scatter(path[0,0], path[0, 1], color='red', marker='*', s=200, zorder=5)
        ax.scatter(path[-1,0], path[-1, 1], color='green', marker='*', s=200, zorder=5)

        # plt.show()
        plt.savefig("plots/path")
    
    def a_star(self, start_point, end_point):

        unexplored = []
        heapq.heappush(unexplored, (0, start_point))  # (cost, point)
        
        # Costs to each node
        g_n = {tuple(start_point): 0}
        
        # Store total costs (cost to neighbor + to goal)
        f_n = {tuple(start_point): np.linalg.norm(start_point - end_point)}
        
        # Store parents to find full path
        parents = {}
        
        # While we haven't explored the whole graph
        while unexplored:

            # Get the node with the lowest cost
            current_f, current_node = heapq.heappop(unexplored)
            
            # Check if goal reached
            if np.array_equal(current_node, end_point):

                # Initialize array to store path
                path = []

                # Get parents
                while tuple(current_node) in parents:
                    path.append(current_node)
                    current_node = parents[tuple(current_node)]
                
                # Add in start
                path.append(start_point)

                # Set right direction
                path.reverse()
                
                # Plot the path
                self.plot_path(path)
                return path
            
            # Iterate through neighbors
            for neigh, weight in self.edges.get(tuple(current_node), []):
                g_neigh = g_n[tuple(current_node)] + weight

                # if neigh costs not initialized or it is better (less than the current cost), update
                if tuple(neigh) not in g_n or g_neigh < g_n[tuple(neigh)]:

                    # update g of the neigh
                    g_n[tuple(neigh)] = g_neigh

                    # Update total cost (f = g + h)
                    f_n[tuple(neigh)] = g_neigh + np.linalg.norm(neigh - end_point)

                    # Add to heap
                    heapq.heappush(unexplored, (f_n[tuple(neigh)], neigh))

                    # Set parent
                    parents[tuple(neigh)] = current_node
        
        return None
    
    # Plot just the graph (for debugging)
    def plot_graph(self):
        fig, ax = plt.subplots()

        for v in self.verticies:
            # ax.scatter(v[0], v[1], color='black')
            ax.scatter(v[1], v[0], color='black')

            all_neigh = self.edges.get(tuple(v), [])
            for neigh, _ in all_neigh:
                x1, y1 = v
                x2, y2 = neigh
                # ax.plot([x1, x2], [y1, y2], color='blue')
                ax.plot([y1, y2], [x1, x2], color='blue')
        
        ax.scatter(self.robot_cell[0], self.robot_cell[1], color='green', zorder=5)
        ax.scatter(self.goal_cell[0], self.goal_cell[1], color='green', zorder=5)

        # plt.show()
        plt.savefig("plots/graph")

    def add_edge(self, node1, node2, weight):
        if node1 not in self.edges:
            self.edges[node1] = [(node2, weight)]
        else:
            self.edges[node1].append((node2, weight))

        if node2 not in self.edges:
            self.edges[node2] = [(node1, weight)]
        else:
            self.edges[node2].append((node1, weight))
    
    # record edges and verticies in the map
    def map_2_graph(self):

        # for i in range(self.map_height):
        #     for j in range(self.map_width):
        for i in np.arange(0, self.map_height, self.cell_size):
            for j in np.arange(0, self.map_width, self.cell_size):

                node = (i, j)

                # if i, j isn't an obstacle
                if 0 <= self.map[i, j] < 10:

                    self.verticies.append(node)

                    # if the cell to the right isn't an obstacle, add an edge
                    if j < self.map_width - self.cell_size and 0 <= self.map[i,j+self.cell_size] < 10:
                        self.add_edge(node, (i,j+self.cell_size), self.cell_size)

                    # if the cell above isn't and obstalce, add an edge
                    if i < self.map_height - self.cell_size and 0 <= self.map[i + self.cell_size, j] < 10:
                        self.add_edge(node, (i + self.cell_size, j), self.cell_size)
                    
                    if i < self.map_height - self.cell_size:
                        
                        # if upper left diagonal is not an obs, add edge
                        if j > 0 and 0 <= self.map[i + self.cell_size, j-self.cell_size] < 10:
                            self.add_edge(node, (i + self.cell_size, j-self.cell_size), self.cell_size * 2**0.5)

                        # if upper right diagonal is not an obs, add edge
                        if j < self.map_width - self.cell_size and 0 <= self.map[i + self.cell_size, j+self.cell_size] < 10:
                            self.add_edge(node, (i + self.cell_size, j+self.cell_size), self.cell_size * 2**0.5)

    def plan_path(self, start_point, end_point, map):

        self.get_logger().info("CONVERTING TO GRAPH")
        self.map_2_graph()

        self.get_logger().info("PLOTTING GRAPH")
        self.plot_graph()
        self.get_logger().info("DONE PLOTTING")

        self.get_logger().info("RUNNING A*")

        self.get_logger().info(f"START = {start_point}")

        self.get_logger().info(f"{type(self.verticies[0][0])}")
        path = self.a_star(np.array([np.int64(start_point[1]), np.int64(start_point[0])]), np.array([np.int64(end_point[1]), np.int64(end_point[0])])) # TODO: I think inidicies should be swapped but this may be error
        
        self.get_logger().info("DONE!!!")

        # self.trajectory.points = path
        
        # self.traj_pub.publish(self.trajectory.toPoseArray())
        # self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
