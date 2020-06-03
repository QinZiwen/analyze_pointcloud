import os
import sys
import numpy as np
import open3d as o3d
import argparse

def show_pcd_from_txt(file_name):
    pcd = []
    color = []
    with open(file_name, 'r') as f:
        for line in f:
            if len(line) < 1 or line[0] == '#':
                continue
            lines = line.strip().split(' ')
            if (len(lines) == 3):
                pcd.append([float(x) for x in lines])
            elif (len(lines) == 6):
                pcd.append([float(x) for x in lines[0:3]])
                color.append([float(x) for x in lines[3:6]])
            else:
                print("file format error, every line size is 3 or 6")
                sys.exit(1)
    
    print('pcd size: ', len(pcd))
    draw_pcd = o3d.geometry.PointCloud()
    draw_pcd.points = o3d.utility.Vector3dVector(pcd)
    if len(color) > 0:
        draw_pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([draw_pcd])

def show_ply(ply_file):
    print("Load a ply point cloud, print it, and render it")
    print(ply_file)
    pcd = o3d.io.read_point_cloud(ply_file, format='xyzn')
    print(pcd.dimension())
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show pointcloud")
    parser.add_argument('--format', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    args = parser.parse_args()

    if args.format == 'txt':
        show_pcd_from_txt(args.file_name)
    elif args.format == 'ply':
        show_ply(args.file_name)
    else:
        print("Invalid format: ", args.format)
