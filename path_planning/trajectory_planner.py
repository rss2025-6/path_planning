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
import cv2

sin = np.sin
cos = np.cos

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.

    To Run On Car:
        1. teleop
        2. ros2 launch path_planning real.launch.xml
        3. ros2 launch racecar_simulator localization_simulate.launch.xml

        Give it an initial pose estimate* and then a goal pose and it should plan a path/move
            add the path topic in rviz (need to run (3.) first to see it) to see the path

    *Note: it's ready for the pose estimate and goal pose if trajectory-planner prints out 
    "Map initialized" in the terminal. It will throw an error if pose/goal are initialized
    but it hasn't received the map
        If it doesn't grab the map, you just need to kill (3.) and run it again

    """

    def __init__(self):
        super().__init__("trajectory_planner")

        # Declare parameters
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        # Get parameters
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        # INITIALIZE SUBSCRIBERS
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

        # INITIALIZE PUBLISHERS
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

        # Initialize trajectory
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # Initialize position variables
        self.goal_coords = None
        self.robot_pose = None
        self.robot_cell = None
        self.goal_cell = None

        # Initialize cell size
        self.cell_size = 2

        # Initialize inflation size
        self.inflation = 17

        # Initialize graph
        self.edges = {}
        self.verticies = []
        self.occupied_verticies = []

        # Flag for plotting
        self.plot_flag = 0

    # Subscribe to map topic and save data
    def map_cb(self, msg):    

        # Convert the map to a numpy array
        self.map = np.array(msg.data)

        # Set unknown as obstacles
        neg_inds = np.where(self.map == -1)
        self.map[neg_inds] = 100
        
        # Save map info
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        # Save map origin
        self.map_origin = msg.info.origin

        x = self.map_origin.orientation.x
        y = self.map_origin.orientation.y
        z = self.map_origin.orientation.z
        w = self.map_origin.orientation.w
        r, p, self.map_yaw = euler_from_quaternion([x, y, z, w])

        # Reshape 1D map to 2D
        self.map = np.reshape(self.map, (self.map_height, self.map_width))

        # Print map info
        self.get_logger().info("MAP INITIALIZED")
        self.get_logger().info(f"SHAPE = {np.shape(self.map)}")
        self.get_logger().info(f"RESOLUTION = {self.map_resolution}")
        self.get_logger().info(f"WIDTH = {self.map_width}")
        self.get_logger().info(f"WIDTH = {self.map_height}")

    def xy_2_uv(self, pose):
        uv = pose[:]
        uv -= np.array([self.map_origin.position.x, self.map_origin.position.y])
        R = np.array([[cos(self.map_yaw), -sin(self.map_yaw)], [sin(self.map_yaw), cos(self.map_yaw)]])
        uv = np.dot(R, uv)
        uv /= self.map_resolution

        self.get_logger().info(f"POSE = {uv}")
        self.get_logger().info(f"POSE ADJ = {uv - uv%self.cell_size}")

        return uv - uv%self.cell_size

    def uv_2_xy(self, pose):

        path_xy = pose[:]
        path_xy *= self.map_resolution
        path_xy = path_xy @ np.linalg.inv(np.array([[cos(self.map_yaw), -sin(self.map_yaw)], [sin(self.map_yaw), cos(self.map_yaw)]]))
        path_xy += np.array([self.map_origin.position.y, self.map_origin.position.x])

        y = np.array([path_xy[:, 0]])
        x = np.array([path_xy[:, 1]])
        path_xy = np.hstack((x.T, y.T))

        return path_xy

    # Record the robot pose
    def pose_cb(self, pose):
        self.robot_pose = np.array([pose.pose.pose.position.x, pose.pose.pose.position.y]) 
        # self.get_logger().info(f"POSITION SET AT {self.robot_pose}")

        self.robot_cell = self.xy_2_uv(np.array([pose.pose.pose.position.x, pose.pose.pose.position.y]))

    # Record the goal coordinates
    def goal_cb(self, msg):
        self.goal_coords = np.array([msg.pose.position.x, msg.pose.position.y])

        self.goal_cell = self.xy_2_uv(np.array([msg.pose.position.x, msg.pose.position.y]))

        self.get_logger().info("GOAL SET")

        self.plan_path(self.robot_cell, self.goal_cell, self.map)

    # Plot full path
    def plot_path(self, path):
        self.get_logger().info("PLOTTING PATH")
        fig, ax = plt.subplots(figsize=(8,6))

        # Iterate through verticies
        for v in self.verticies:

            # Plot each vertex
            ax.scatter(v[1], v[0], color='dodgerblue', marker='o', s=3)

            # Get neighbors
            all_neigh = self.edges.get(tuple(v), [])

            # Plot edges between neighbors
            for neigh, _ in all_neigh:
                x1, y1 = v
                x2, y2 = neigh
                ax.plot([y1, y2], [x1, x2], color='dodgerblue', lw=1)

        ax.plot(path[:, 1], path[:, 0], color='black', lw = 1)

        ax.scatter(path[0,1], path[0, 0], color='black', marker='*', s=30, zorder=5)
        ax.scatter(path[-1,1], path[-1, 0], color='green', marker='*', s=30, zorder=5)

        ax.set_aspect('equal')

        # plt.show()
        plt.savefig("plots/path")
        self.get_logger().info("PLOT SAVED TO plots/path")
    
    # Given a start point and end point, finds a path through the graph
    def a_star(self, start_point, end_point):

        unexploblack = []
        heapq.heappush(unexploblack, (0, start_point))  # (cost, point)
        
        # Costs to each node
        g_n = {tuple(start_point): 0}
        
        # Store total costs (cost to neighbor + to goal)
        f_n = {tuple(start_point): np.linalg.norm(start_point - end_point)}
        
        # Store parents to find full path
        parents = {}
        
        # While we haven't exploblack the whole graph
        while unexploblack:

            # Get the node with the lowest cost
            current_f, current_node = heapq.heappop(unexploblack)
            
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

                # Convert to np
                path = np.array(path)
                
                # Plot the path
                if self.plot_flag:
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
                    heapq.heappush(unexploblack, (f_n[tuple(neigh)], neigh))

                    # Set parent
                    parents[tuple(neigh)] = current_node
        
        return None
    
    # Plot just the graph
    def plot_graph(self):
        fig, ax = plt.subplots(figsize=(8,6))

        for v in self.verticies:
            ax.scatter(v[1], v[0], color='dodgerblue', marker='o', s=3)

            all_neigh = self.edges.get(tuple(v), [])
            for neigh, _ in all_neigh:
                x1, y1 = v
                x2, y2 = neigh
                ax.plot([y1, y2], [x1, x2], color='dodgerblue', lw=1)
        
        ax.scatter(self.robot_cell[0], self.robot_cell[1], color='green', marker='o', s=5, zorder=5)
        ax.scatter(self.goal_cell[0], self.goal_cell[1], color='green', marker='o', s=5, zorder=5)

        ax.set_aspect('equal')

        plt.savefig("plots/graph")
        self.get_logger().info("PLOT SAVED TO plots/graph")

    # Plot in xy space
    def plot_xy(self, path):
        fig, ax = plt.subplots(figsize=(8,6))

        for v in self.verticies:
            arr = np.array([[v[0], v[1]]])
            arr = arr.astype(np.float64)
            v_xy = self.uv_2_xy(arr)
            ax.scatter(v_xy[0,0], v_xy[0,1], color = 'dodgerblue', marker = 'o', s=3)

            all_neigh = self.edges.get(tuple(v), [])
            for neigh, _ in all_neigh:
                x1, y1 = v_xy[0]

                arr = np.array([[neigh[0], neigh[1]]])
                arr = arr.astype(np.float64)
                neigh_xy = self.uv_2_xy(arr)
                x2, y2 = neigh_xy[0]
                ax.plot([x1, x2], [y1, y2], color='dodgerblue', lw=1)

        ax.plot(path[:, 0], path[:, 1], color='black', lw = 1)

        ax.scatter(path[0,0], path[0, 1], color='black', marker='*', s=30, zorder=5)
        ax.scatter(path[-1,0], path[-1, 1], color='green', marker='*', s=30, zorder=5)

        ax.set_aspect('equal')

        # plt.show()
        plt.savefig("plots/graph_xy")
        self.get_logger().info("PLOT SAVED TO plots/graph_xy")

    # Add edges to the graph like {node: [(neighbor), weight]}
    # Adds edge both from node 1 to node 2 and from node 2 to node 1
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
        
        if self.plot_flag:
            for i in range(self.map_height):
                for j in range(self.map_width):
                    if self.map[i, j] >= 10:
                        self.occupied_verticies.append([i, j])

    # Inflates obstacles by self.inflation
    def dilate_map(self):
        # Create kernel for inflation
        kernel = np.ones((self.inflation, self.inflation), np.uint8)

        # Inflate the map
        self.map = cv2.dilate(self.map.astype(np.uint8), kernel, iterations=1)

    # Plot occupied spaces
    def plot_plain_map(self):
        fig, ax = plt.subplots(figsize=(8,6))

        self.occupied_verticies = np.array(self.occupied_verticies)
        ax.scatter(self.occupied_verticies[:, 1], self.occupied_verticies[:, 0], color = "black", marker = 'o', s=1)

        ax.set_aspect("equal")

        plt.savefig("plots/map_outline")
        self.get_logger().info("PLOT SAVED TO plots/map_outline")


    def plan_path(self, start_point, end_point, map):

        # Inflate the obstacles to plan farther from wall
        self.get_logger().info("Dilating Map")
        self.dilate_map()

        # Convert the map to a graph
        self.get_logger().info("CONVERTING TO GRAPH")
        self.map_2_graph()

        # Plot if we are plotting
        if self.plot_flag:
            # Plot the plain map
            self.get_logger().info("PLOTTING Plain MAP...")
            self.plot_plain_map()

            # Plot the graph representation of the map
            self.get_logger().info("PLOTTING GRAPH")
            self.plot_graph()
            self.get_logger().info("DONE PLOTTING")

        # Run A* to plan a path from start to end point
        self.get_logger().info("RUNNING A*")
        path_uv = self.a_star(np.array([np.int64(start_point[1]), np.int64(start_point[0])]), np.array([np.int64(end_point[1]), np.int64(end_point[0])])) # TODO: I think inidicies should be swapped but this may be error
        self.get_logger().info("A* DONE!!!")

        # Convert uv space to xy space
        path_xy = self.uv_2_xy(path_uv.astype(np.float64))

        # Add in actual start and goal
        path_xy = np.vstack((self.robot_pose, path_xy[1:,:]))
        path_xy = np.vstack((path_xy[:-1,:], self.goal_coords))

        # Plot if we are plotting
        if self.plot_flag:
            # Plot path and graph in xy space
            self.get_logger().info("PLOTTING XY")
            self.plot_xy(path_xy)
            self.get_logger().info("DONE PLOTTING")

        # Publish the trajectory
        self.trajectory.points = path_xy
        
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
