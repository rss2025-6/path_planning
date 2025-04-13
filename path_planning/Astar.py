import numpy as np
import heapq
from math import sqrt
# assuming an integer coordinate 2D bounding grid g
def node(pose, g, h, parent) -> dict:
    # make a new node
    return {'pose': pose, 'g': g, 'h': h, 'f': g+h, 'parent': parent}

def dist(a: tuple[int, int], b: tuple[int,int]) -> float:
    # euclidean distance
    x,y= a
    x1,y1=b
    return sqrt((x-x1)**2+(y-y1)**2)

def adjacent(cspace: np.ndarray, pose: tuple[int, int]) -> list[tuple[int,int]]:
    # valid neighbors
    x,y=pose
    adj= [(x,y+1),(x,y-1),(x-1,y),(x+1,y),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)]
    return [(kx,ky) for kx,ky in adj if 0<=kx<cspace.shape[0] and 0<=ky<cspace.shape[1] and cspace[kx,ky]==0]
    
def Astar(cspace: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int,int]]
    s_node=node(pose=start, g=0,h=dist(start,goal))
    to_check=[(s_node['f'], start)] # pq for checking 
    to_check_dict = {start: s_node} 
    done = set() #set of position pairs to ignore

    while (to_check):
        #more to check
    fm, current_pose = heapq.heappop(to_check)
    current=to_check_dict[current_pose]
    if current_pose == goal: # retrace 
        path=[]
        while current is not None:
            path.append(current['pose'])
            current = current['parent']
            return path[::-1]
    
    done.add(current_pose)
    neighbors=adjacent(cspace, current_pose)
    for nodes in neighbors: # nodes are (x,y) position pairs 
        if not (nodes in done): 
            cost = current_pose['g']+dist(current_pose, nodes)
            if not (nodes in to_check_dict):
                neighbor= node(nodes, cost, dist(nodes,goal), current_pos)
                heapq.heappush(to_check, (neighbor['f'],nodes))
                #to_check.append(neighbor)
                to_check_dict[nodes]=neighbor
            else:
                if (cost<to_check_dict[nodes]['g']):
                    neighbor=to_check_dict[nodes]
                    neighbor['f']=cost+neighbor['h']
                    neighbor[parent]=current
                    neighbor['g']=cost
    return []







        



