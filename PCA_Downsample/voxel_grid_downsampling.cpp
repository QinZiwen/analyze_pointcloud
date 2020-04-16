#include "PCA_Downsample/voxel_grid_downsampling.h"

#include <cmath>
#include <algorithm>
#include <fstream>

bool VoxelGridDownsampling::set_data(const std::string& pts_file) {
    _pcd = read_pointcloud_from_file(pts_file);
}

bool VoxelGridDownsampling::downsampling(int grid_size) {
    Eigen::Vector3d min = _pcd.rowwise().min();
    Eigen::Vector3d max = _pcd.rowwise().max();

    Eigen::Vector3d dim = (max - min) / grid_size;

    std::vector<std::pair<int, Eigen::Vector3d>> _downsample;
    for (size_t i = 0; i < _pcd.cols(); ++i) {
        Eigen::Vector3d pts = _pcd.cos(i);
        int h_x = floor((pts(0) - min(0)) / grid_size);
        int h_y = floor((pts(1) - min(1)) / grid_size);
        int h_z = floor((pts(2) - min(2)) / grid_size);

        int h = h_x + dim(0) * h_y + dim(0) * dim(1) * h_z;
        _downsample.emplace_back(std::make_pair(h, pts));
    }

    std::sort(_downsample.begin(), _downsample.end(), [] (const std::pair<int, Eigen::Vector3d>& p1, const std::pair<int, Eigen::Vector3d>& p2) -> bool {
        return p1.first < p2.first;
    });

    // select_point
    select_centroid(_downsample);
}

bool VoxelGridDownsampling::select_centroid(std::vector<std::pair<int, Eigen::Vector3d>>& downsample) {
    std::vector<std::pair<int, Eigen::Vector3d>> res;
    std::vector<std::pair<int, Eigen::Vector3d>> grid_pts;
    for (size_t i = 0; i < dowsample.size(); ++i) {
        if (grid_pts.size() == 0) {
            grid_pts.emplace_back(downsample[i]);
        } else if (grid_pts[grid_pts.size() - 1].first != downsample[i].first) {
            // find_centroid
            double cx = 0, cy = 0, cz = 0;
            for (size_t j = 0; j < grid_pts.size(); ++j) {
                cx += grid_pts[j].second[0];
                cy += grid_pts[j].second[1];
                cz += grid_pts[j].second[2];
            }

            cx /= grid_pts.size();
            cy /= grid_pts.size();
            cz /= grid_pts.size();

            res.emplace_back(std::make_pair(grid_pts[0].first, Eigen::Vector3d(cx, cy, cz)));
            // clear and push new data

            grid_pts.clear();
            grid_pts.emplace_back(downsample[i]);
        } else {
            grid_pts.emplace_back(downsample[i]);
        }
    }

    downsample.swap(res);
    return true;
}

bool VoxelGridDownsampling::select_random(std::vector<std::pair<int, Eigen::Vector3d>>& downsample) {
    return true;
}

bool save_to_file(const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    for (cons auto& d : _downsample) {
        ofs << d.second[0] << " " << d.second[1] << " " << d.second[2] << std::endl;
    }

    ofs.close();
    return true;
}