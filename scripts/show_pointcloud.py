import os
import sys
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    ply_file = sys.argv[1]
    print("Load a ply point cloud, print it, and render it")
    print(ply_file)
    pcd = o3d.io.read_point_cloud(ply_file, format='xyz')
    print(pcd)
    o3d.visualization.draw_geometries([pcd])


# print("Let's draw a cubic using o3d.geometry.LineSet.")
# points = [
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 1, 0],
#     [0, 0, 1],
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1],
# ]
# lines = [
#     [0, 1],
#     [0, 2],
#     [1, 3],
#     [2, 3],
#     [4, 5],
#     [4, 6],
#     [5, 7],
#     [6, 7],
#     [0, 4],
#     [1, 5],
#     [2, 6],
#     [3, 7],
# ]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(points),
#     lines=o3d.utility.Vector2iVector(lines),
# )
# line_set.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([line_set])