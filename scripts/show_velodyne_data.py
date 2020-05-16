import os
import numpy as np
import struct
import open3d
import sys

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def main(root_dir):
    filename=os.listdir(root_dir)
    file_number=len(filename)

    pcd=open3d.open3d.geometry.PointCloud()

    for i in range(file_number):
        path=os.path.join(root_dir, filename[i])
        print(path)
        example=read_bin_velodyne(path)
        # From numpy to Open3D
        pcd.points= open3d.open3d.utility.Vector3dVector(example)
        open3d.open3d.visualization.draw_geometries([pcd])

if __name__=="__main__":
    main(sys.argv[1])
