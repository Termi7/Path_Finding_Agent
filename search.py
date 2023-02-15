import math

import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.animation as animation
from matplotlib.path import Path

from utils import *
from grid import *


def gen_polygons(worldfilepath):
    polygons = []
    with open(worldfilepath, "r") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for line in lines:
            polygon = []
            pts = line.split(';')
            for pt in pts:
                xy = pt.split(',')
                polygon.append(Point(int(xy[0]), int(xy[1])))
            polygons.append(polygon)
    return polygons


def a_star(start, end, p1, p2):
    priorityq = PriorityQueue()
    visited = []
    path = []
    priorityq.push(start, 0)

    while not priorityq.isEmpty():
        curr_node = priorityq.pop()

        if curr_node.x == end.x and curr_node.y == end.y:
            current = curr_node
            while current is not None:
                path.append(Point(current.x, current.y))
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Up, right, down, and left

            # Get node position
            node_position = [curr_node.x + new_position[0], curr_node.y + new_position[1]]

            # Make sure within range
            if node_position[0] >= 50 or node_position[0] < 0 or node_position[1] >= 50 or node_position[1] < 0:
                continue

            # Check if node is within the polygon
            inside_polygon = False
            for p in p1:
                if p.contains_point(node_position, radius=-1):
                    inside_polygon = True
                    break
            if inside_polygon:
                continue

            # Make sure not visited
            if node_position in visited:
                continue

            # Create new node
            new_node = Point(node_position[0], node_position[1], curr_node)

            # Append
            children.append(new_node)
            visited.append(node_position)
            action_cost =0

        # Add children to the heap with estimated cost
        for child in children:
            inside_polygon = False
            for p in p2:
                if p.contains_point([child.x, child.y], radius=-1):
                    inside_polygon = True
                break
            if inside_polygon:
                action_cost = 1.5 + action_cost
            else:
                action_cost = 1 + action_cost
            le = child.to_tuple()
            en = end.to_tuple()
            estimated_cost = math.sqrt((le[0] - en[0]) ** 2 + (le[1] - en[1]) ** 2)

            priorityq.update(child, estimated_cost + action_cost)


def gbfs(start, end, p1, p2):
    priorityq = PriorityQueue()
    visited = []
    path = []
    priorityq.push(start, 0)
    while not priorityq.isEmpty():
        curr_node = priorityq.pop()

        if curr_node.x == end.x and curr_node.y == end.y:
            current = curr_node
            while current is not None:
                path.append(Point(current.x, current.y))
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Up, right, down, and left

            # Get node position
            node_position = [curr_node.x + new_position[0], curr_node.y + new_position[1]]

            # Make sure within range
            if node_position[0] >= 50 or node_position[0] < 0 or node_position[1] >= 50 or node_position[1] < 0:
                continue

            inside_polygon = False
            for p in p1:
                if p.contains_point(node_position, radius=-1):
                    inside_polygon = True
                    break
            if inside_polygon:
                continue

            # Make sure not visited
            if node_position in visited:
                continue

            # Create new node
            new_node = Point(node_position[0], node_position[1], curr_node)

            # Append
            children.append(new_node)
            visited.append(node_position)

        # Add children to the heap with estimated cost
        for child in children:
            le = child.to_tuple()
            en = end.to_tuple()
            estimated_cost = math.sqrt((le[0] - en[0]) ** 2 + (le[1] - en[1]) ** 2)

            priorityq.update(child, estimated_cost)
    return None


def dfs(start, end, p1, p2):
    stack = Stack()
    visited = []
    path = []

    # Create Paths from epolygons
    # paths = [Path(np.asarray(polygon)) for polygon in epolygons]

    stack.push(start)
    le = start.to_tuple()
    visited.append(le)

    while not stack.isEmpty():
        curr_node = stack.pop()


        if curr_node.x == end.x and curr_node.y == end.y:
            current = curr_node
            while current is not None:
                path.append(Point(current.x, current.y))
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Up, right, down, and left
            # print(new_position)

            # Get node position
            node_position = [curr_node.x + new_position[0], curr_node.y + new_position[1]]


            # Make sure within range
            if node_position[0] >= 50 or node_position[0] < 0 or node_position[1] >= 50 or node_position[1] < 0:
                continue

            # Make sure not visited
            if node_position in visited:
                continue

            inside_polygon = False
            for p in p1:
                if p.contains_point(node_position, radius=-1):
                    inside_polygon = True
                    break
            if inside_polygon:
                continue

            # print(node_position)
            # Create new node
            new_node = Point(node_position[0], node_position[1])
            new_node.parent = curr_node

            children.append(new_node)
            stack.push(new_node)
            visited.append(node_position)


        # Add children to the stack
        # for child in children[::-1]:
        #     stack.push(child)
    return None


def bfs(start, end, p1, p2):

    queue = Queue()
    visited = []
    path = []

    queue.push(start)

    while not queue.isEmpty():

        # print(path)
        cur_node = queue.pop()
        if cur_node.x == end.x and cur_node.y == end.y:
            current = cur_node
            while current is not None:
                path.append(Point(current.x, current.y))
                current = current.parent
            return path[::-1]  # Return reversed path

        children = []

        for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Up, right, down, and left

            # Get node position
            node_position = [cur_node.x + new_position[0], cur_node.y + new_position[1]]

            # Make sure within range
            if node_position[0] >= 50 or node_position[0] < 0 or node_position[1] >= 50 or node_position[1] < 0:
                continue

            # Make sure not visited
            if node_position in visited:
                continue

            inside_polygon = False
            for p in p1:
                if p.contains_point(node_position, radius=-1):
                    inside_polygon = True
                    break
            if inside_polygon:
                continue



            # Create new node
            new_node = Point(node_position[0], node_position[1])
            new_node.parent = cur_node

            # Append
            children.append(new_node)
            # queue.push(new_node)
            visited.append(node_position)

        for child in children:
            queue.push(child)

    return None


if __name__ == "__main__":
    epolygons = gen_polygons('TestingGrid/world1_enclosures.txt')
    tpolygons = gen_polygons('TestingGrid/world1_turfs.txt')
    # print(tpolygons)
    # print(tpolygons)

    # source = Point(24, 17)
    # dest = Point(28, 20)
    source = Point(8, 10)
    dest = Point(43, 45)

    fig, ax = draw_board()
    draw_grids(ax)
    draw_source(ax, source.x, source.y)  # source point
    draw_dest(ax, dest.x, dest.y)  # destination point

    # Draw enclosure polygons
    for polygon in epolygons:
        for p in polygon:
            draw_point(ax, p.x, p.y)
    for polygon in epolygons:
        for i in range(0, len(polygon)):
            draw_line(ax, [polygon[i].x, polygon[(i + 1) % len(polygon)].x],
                      [polygon[i].y, polygon[(i + 1) % len(polygon)].y])

    # Draw turf polygons
    for polygon in tpolygons:
        for p in polygon:
            draw_green_point(ax, p.x, p.y)
    for polygon in tpolygons:
        for i in range(0, len(polygon)):
            draw_green_line(ax, [polygon[i].x, polygon[(i + 1) % len(polygon)].x],
                            [polygon[i].y, polygon[(i + 1) % len(polygon)].y])

    #### Here call your search to compute and collect res_path
    # polygonEPath = []

    # s1 = (sum(epolygons, []))
    #
    # vertices = []
    # for point in s1:
    #     vertices.append([point.x, point.y])
    # vertices = np.asarray(vertices, float)
    #
    # p1 = Path(vertices)
    #
    #
    # # for s in polygonEPath:
    # #     print(s)
    # # print(polygonEPath)
    #
    # ####
    # vertices = []
    # s2 = (sum(tpolygons, []))
    # for points in s2:
    #     vertices.append([points.x, points.y])
    # vertices2 = np.asarray(vertices, float)
    # p2 = Path(vertices2)


    # for p in epolygons:
    #     print(p)
    #     vertice=[]
    polygonEPath = []
    for polygon in epolygons:
        vertice=[]
        for point in polygon:
            vertice.append([point.x, point.y])
        vertices = np.asarray(vertice, float)
        p1 = Path(vertices)
        polygonEPath.append(p1)

    polygonTPath = []
    for polygon in tpolygons:
        vertice = []
        for point in polygon:
            vertice.append([point.x, point.y])
        vertices = np.asarray(vertice, float)
        p1 = Path(vertices)
        polygonTPath.append(p1)






    print("Weclome to the pathfinding Maze game.")
    print("Select and option below")
    print("1- DFS")
    print("2- BSF")
    print("3- Greedy Best First Search")
    print("4- A* search")
    # x= input("What pathfinding algorithm you like to perform")
    #
    # choice = input("What's the programming language you want to learn? ")
    res_path= []

    while True:
        choice = int(input("What pathfinding algorithm you like to perform: "))
        if  1 <= int(choice) <= 4:
            break
        else:
            print("Invalid input. Please enter a number from 1 to 4: ")

    match choice:
        case 1:
            res_path = dfs(source, dest, polygonEPath, polygonTPath)

        case 2:
            res_path = bfs(source, dest, polygonEPath,  polygonTPath)

        case 3:
            res_path = gbfs(source, dest, polygonEPath,  polygonTPath)

        case 4:
            res_path = a_star(source, dest, polygonEPath,  polygonTPath)



    # res_path = bfs(source, dest, p1, p2)

    # res_path = dfs(source, dest, p1, p2)

    # res_path = gbfs(source, dest, p1, p2)

    # res_path = a_star(source, dest, p1, p2)

    # print(res_path)

    # res_path = [Point(24,17), Point(25,17), Point(26,17), Point(27,17),
    #             Point(28,17), Point(28,18), Point(28,19), Point(28,20)]

    if res_path is not None:
        for i in range(len(res_path) - 1):
            draw_result_line(ax, [res_path[i].x, res_path[i + 1].x], [res_path[i].y, res_path[i + 1].y])
            # plt.pause(0.1)

    plt.show()
