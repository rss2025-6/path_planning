import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Float32
import numpy as np
from .utils import LineTrajectory

class TreeNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0
class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
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

        ###Start RRT###
        self.map_graph = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.root_position = None
        self.goal_position = None
        self.tolerance = 1.0
        self.tree = None

    def map_cb(self, msg):
        # create a graph of valid positions from map
        self.get_logger().info(f"Received map: {msg.info.width}x{msg.info.height} resolution: {msg.info.resolution}")
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        cells = msg.data
        self.map_graph = self.create_graph(self, self.map_width, self.map_height, cells, self.map_resolution)
        
    # Proceed with your logic


    def pose_cb(self, pose):
        # get robot pose
        self.current_x = pose.pose.position.x
        self.current_y = pose.pose.position.y
        self.current_orient = pose.pose.orientation
        x = self.current_orient.x
        y = self.current_orient.y
        z = self.current_orient.z
        w = self.current_orient.w
        r, p, self.true_yaw = euler_from_quaternion([x, y, z, w])
        self.root_position = TreeNode(self.current_x, self.current_y)

    def goal_cb(self, msg):
        # get goal pose
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_position = TreeNode(self.goal_x, self.goal_y)

    def plan_path(self, start_point, end_point, map):
        path = self.rrt(start_point, end_point, map)
        for node in path:
            self.trajectory.addPoint((node.x, node.y))
        # msg = PoseArray()
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    ### RRT Helper Functions ###
    def rrt(self, start, goal, map):
        tree = [start]
        while self.goal_condition(tree[-1], self.tolerance) is False:
            random_node = self.sample(map)
            nearest_node = self.get_nearest_node(random_node,tree)
            random_node.parent = nearest_node
            tree.append(random_node)

        last_node = tree[-1]
        current_node = last_node
        path = [current_node]

        while current_node.parent is not None:
            parent = current_node.parent
            path.append(parent)
            current_node = parent

        return path
    
    def sample(self, map):
        random_node = np.random.choice(map.vertices, 1)
        x = random_node[0] * self.map_resolution
        y = random_node[1] * self.map_resolution
        return TreeNode(x,y)
        # return random_node

    
    def get_nearest_node(self, new_node, tree):
        min_node = None
        min_dist = np.inf
        for node in tree:
            dist = np.linalg.norm([node.x-new_node.x, node.y-new_node.y])
            if dist < min_dist:
                min_node = node
                min_dist = dist
       
        return min_node
    
    def create_graph(self, width, height, cells, map_resolution): # might need to add parents?
        # graph = {}
        vertices = set()
        edges = set()
        for r in range(height):
            for c in range(width):
                index = r*height + c
                # x = map_resolution * r
                # y = map_resolution * c
                probability = cells[index]
                # i f 0 < probability < 50: # might need to adjust, only want accessible positions in graph
                if 0 < probability < 50:
                    vertices.add((r,c))
                    adj = {(r+i, c+j) for i in (-1,0,1) for j in (-1,0,1) if 0<=r+i<height and 0<=c+j<width and 0<cells[(r+i)*height +(c+j)]<50}
                    edges.update(adj)
                # valid_node = TreeNode(x, y)
                # valid_node.cost = probability # may help with choosing points later by leaning towards more probabaly open spots
                # graph.append(valid_node)
        return Graph(vertices, edges)
    
    def goal_condition(self, current_node, tolerance): # ✅
        # check if current node is within acceptable tolerance position of goal
        x = current_node.x
        y = current_node.y
        goal_x = self.goal_position.x
        goal_y = self.goal_position.y
        return np.linalg.norm([x-goal_x, y-goal_y]) < tolerance


    ### End RRT Helper Functions
        

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
