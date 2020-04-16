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
