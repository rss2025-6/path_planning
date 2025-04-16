import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion
import matplotlib.pyplot as plt
import numpy as np
from .utils import LineTrajectory
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


        ### RRT variables ###
        self.map = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.map_yaw = None

        self.cell_size = 5

        self.robot_pose = None
        self.robot_cell = None
        self.goal_cell = None

        self.map_nodes = []
        self.map_edges = {}

        #####################
    def map_cb(self, msg):

        # Convert the map to a numpy array
        self.map = np.array(msg.data)
        
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
        self.get_logger().info(f"WIDTH = {self.map_width}")
        self.get_logger().info(f"WIDTH = {self.map_height}")

    def pose_cb(self, pose):
        self.robot_pose = np.array([pose.pose.pose.position.x, pose.pose.pose.position.y]) 
        # self.get_logger().info(f"POSITION SET AT {self.robot_pose}")
        self.robot_cell = self.xy_2_uv(np.array([pose.pose.pose.position.x, pose.pose.pose.position.y]))

    def goal_cb(self, msg):
        self.goal_coords = np.array([msg.pose.position.x, msg.pose.position.y])
        self.goal_cell = self.xy_2_uv(np.array([msg.pose.position.x, msg.pose.position.y]))
        self.get_logger().info("GOAL SET")
        start = (self.robot_cell[0], self.robot_cell[1])
        end = (self.goal_cell[0], self.goal_cell[1])
        self.plan_path(start, end, self.map)

    def plan_path(self, start_point, end_point, map):
        # Convert the map to a graph
        self.get_logger().info("CONVERTING TO GRAPH")
        self.map_2_graph()
        # self.get_logger().info("PLOTTING GRAPH")
        # self.plot_graph()
        self.get_logger().info("RUNNING RRT")
        path_uv = self.rrt(start_point, end_point) # TODO: I think inidicies should be swapped but this may be error
        self.get_logger().info("RRT DONE!!!")
        self.get_logger().info(f"PATH: {path_uv}")
        # path_xy = self.uv_2_xy(path_uv.astype(np.float64))

        # Add in actual start and goal
        path_uv_array = np.array(path_uv)

        # Ensure the array is of type float64
        path_uv_array = path_uv_array.astype(np.float64)

        # Now you can pass the array to uv_2_xy
        path_xy = self.uv_2_xy(path_uv_array)
        path_xy = np.vstack((self.robot_pose, path_xy[1:,:]))
        path_xy = np.vstack((path_xy[:-1,:], self.goal_coords))

        self.trajectory.points = path_xy

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    ### RRT Functions ###
    def rrt(self, start_node, goal_node): # run rrt in uv coords
        tree_nodes = [start_node]
        tree_edges = {}
        max_iterations = 1000  # Add a limit to prevent an infinite loop
        iterations = 0
        arrived = False

        while not arrived:
            iterations += 1
            random_node = self.sample()
            self.get_logger().info(f'Iteration {iterations}: Random Node: {random_node}')
            
            nearest_node = self.nearest(random_node, tree_nodes)
            self.get_logger().info(f'Nearest Node: {nearest_node}')
            
            new_node = self.steer(nearest_node, random_node)
            self.get_logger().info(f'New Node: {new_node}')
            
            if new_node != nearest_node:
                tree_nodes.append(new_node)
                self.add_edge(nearest_node, new_node, tree_edges)
            
            # Check if we've reached the goal
            distance_to_goal = np.linalg.norm(np.array(new_node) - np.array(goal_node))
            self.get_logger().info(f"Distance to Goal: {distance_to_goal}")

            if distance_to_goal < 400:  # Adjust the tolerance based on your requirements
                arrived = True

        return self.build_path(tree_nodes, tree_edges)

    

    
    def goal_condition(self, current_node, goal_node, tolerance = 5): # checks if current node is within a certain tolerance
        self.get_logger().info(f'dist: {np.linalg.norm(np.array(current_node) - np.array(goal_node))}')
        return np.linalg.norm(np.array(current_node) - np.array(goal_node)) < tolerance    
    def build_path(self, nodes, edges):
        self.get_logger().info(f"NODE: {len(nodes)}")
        path = [nodes[-1]]  # Start with the goal node
        current_node = nodes[-1]
        while current_node != nodes[0]:
            for neighbor in edges[current_node]:
                if neighbor not in path:  # Avoid cycles
                    path.insert(0, neighbor)
                    current_node = neighbor
                    break
        return path

    def steer(self, from_node, to_node):
        theta = np.arctan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        
        new_node = (from_node[0] + 5 * cos(theta),
                    from_node[1] + 5 * sin(theta))
        
        new_node_grid = (round(new_node[0] / self.cell_size) * self.cell_size,
                        round(new_node[1] / self.cell_size) * self.cell_size)
        
        if self.is_free(new_node_grid):
            return new_node_grid  # Return the new node if it's free
        else:
            return from_node  # If not free, return the original node (no movement)

    
    def is_free(self, node): # see if node is a free space, -1 --> occupied, 0 --> unoccupied, might be redundant since map_graph only has 
            return node in self.map_nodes
    def sample(self): # choose a random sample from the graph
        index = np.random.choice(range(0,len(self.map_nodes)))
        sample = self.map_nodes[index]
        return sample
    def nearest(self, random_node, tree_nodes): # find nearest node that exists in the current tree; note tree should start with initial position of robot
        min_node = None
        min_dist = np.inf
        for node in tree_nodes:
            dist = np.linalg.norm([node[0] - random_node[0], node[1] - random_node[1]])
            if dist < min_dist:
                min_node = node
                min_dist = dist
        return min_node
    def map_2_graph(self):
        for i in np.arange(0, self.map_height, self.cell_size):
            for j in np.arange(0, self.map_width, self.cell_size):
                node = (i, j)
                # if i, j isn't an obstacle
                if 0 <= self.map[i, j] < 10:
                    self.map_nodes.append(node)

                    # if the cell to the right isn't an obstacle, add an edge
                    if j < self.map_width - self.cell_size and 0 <= self.map[i,j+self.cell_size] < 10:
                        self.add_edge(node, (i,j+self.cell_size), self.map_edges)

                    # if the cell above isn't and obstalce, add an edge
                    if i < self.map_height - self.cell_size and 0 <= self.map[i + self.cell_size, j] < 10:
                        self.add_edge(node, (i + self.cell_size, j), self.map_edges)
                    
                    if i < self.map_height - self.cell_size:
                        
                        # if upper left diagonal is not an obs, add edge
                        if j > 0 and 0 <= self.map[i + self.cell_size, j-self.cell_size] < 10:
                            self.add_edge(node, (i + self.cell_size, j-self.cell_size), self.map_edges)

                        # if upper right diagonal is not an obs, add edge
                        if j < self.map_width - self.cell_size and 0 <= self.map[i + self.cell_size, j+self.cell_size] < 10:
                            self.add_edge(node, (i + self.cell_size, j+self.cell_size), self.map_edges)
    def uv_2_xy(self, pose):
        path_xy = pose[:]
        path_xy *= self.map_resolution
        path_xy = path_xy @ np.linalg.inv(np.array([[cos(self.map_yaw), -sin(self.map_yaw)], [sin(self.map_yaw), cos(self.map_yaw)]]))
        path_xy += np.array([self.map_origin.position.y, self.map_origin.position.x])
        y = np.array([path_xy[:, 0]])
        x = np.array([path_xy[:, 1]])
        path_xy = np.hstack((x.T, y.T))
        return path_xy
    def xy_2_uv(self, pose):
        uv = pose[:]
        uv -= np.array([self.map_origin.position.x, self.map_origin.position.y])
        R = np.array([[cos(self.map_yaw), -sin(self.map_yaw)], [sin(self.map_yaw), cos(self.map_yaw)]])
        uv = np.dot(R, uv)
        uv /= self.map_resolution

        self.get_logger().info(f"POSE = {uv}")
        self.get_logger().info(f"POSE ADJ = {uv - uv%self.cell_size}")

        return uv - uv%self.cell_size
    def add_edge(self, from_node, to_node, edges): # add bidirectional edge to edge dictionary
        if from_node not in edges:
            edges[from_node] = [from_node]
        else:
            edges[from_node].append(to_node)
        if to_node not in edges:
            edges[to_node] = [from_node]
        else:
            edges[to_node].append(from_node)


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()