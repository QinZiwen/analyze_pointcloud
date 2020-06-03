#include "PCA_Downsample/voxel_grid_downsampling.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <cmath>

bool VoxelGridDownsampling::set_data(const std::string& pts_file) {
    _pcd = Utils::read_pointcloud_from_file(pts_file);
    std::cout << "point cloud size: " << _pcd.cols() << std::endl;
    return true;
}

bool VoxelGridDownsampling::set_data(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& data) {
    _pcd = data;
    return true;
}

bool VoxelGridDownsampling::downsampling(double grid_size, SelectPointsMethod select_pts_method) {
    Eigen::Vector3d min = _pcd.rowwise().minCoeff();
    Eigen::Vector3d max = _pcd.rowwise().maxCoeff();
    std::cout << "min: " << min.transpose() << std::endl;
    std::cout << "max: " << max.transpose() << std::endl;

    Eigen::Vector3d dim = (max - min) / grid_size;
    std::cout << "dim: " << dim.transpose() << std::endl;

    for (size_t i = 0; i < _pcd.cols(); ++i) {
        Eigen::Vector3d pts = _pcd.col(i);
        int h_x = floor((pts(0) - min(0)) / grid_size);
        int h_y = floor((pts(1) - min(1)) / grid_size);
        int h_z = floor((pts(2) - min(2)) / grid_size);

        int h = h_x + dim(0) * h_y + dim(0) * dim(1) * h_z;
        _downsample.emplace_back(std::make_pair(h, pts));
    }
    std::cout << "_downsample size: " << _downsample.size() << std::endl;

    std::sort(_downsample.begin(), _downsample.end(), [] (const std::pair<int, Eigen::Vector3d>& p1, const std::pair<int, Eigen::Vector3d>& p2) -> bool {
        return p1.first < p2.first;
    });

    // select_point
    switch (select_pts_method) {
    case SelectPointsMethod::CENTROID:
        select_centroid(_downsample);
        break;
    case SelectPointsMethod::RANDOM:
        select_random(_downsample);
        break;
    default:
        std::cerr << "invalid select points method" << std::endl;
        return false;
    }
    std::cout << "_downsample size after select: " << _downsample.size() << std::endl;
    return true;
}

bool VoxelGridDownsampling::select_centroid(std::vector<std::pair<int, Eigen::Vector3d>>& downsample) {
    std::vector<std::pair<int, Eigen::Vector3d>> res;
    std::vector<std::pair<int, Eigen::Vector3d>> grid_pts;
    for (size_t i = 0; i < downsample.size(); ++i) {
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
    std::vector<std::pair<int, Eigen::Vector3d>> res;
    std::vector<std::pair<int, Eigen::Vector3d>> grid_pts;
    std::srand(time(0));
    for (size_t i = 0; i < downsample.size(); ++i) {
        if (grid_pts.size() == 0) {
            grid_pts.emplace_back(downsample[i]);
        } else if (grid_pts[grid_pts.size() - 1].first != downsample[i].first) {
            int rand_id = std::rand() % grid_pts.size();
            Eigen::Vector3d rand_pts;
            rand_pts << grid_pts[rand_id].second(0),
                        grid_pts[rand_id].second(1),
                        grid_pts[rand_id].second(2);
            res.emplace_back(std::make_pair(grid_pts[0].first, rand_pts));
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

bool VoxelGridDownsampling::save_to_file(const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    for (const auto& d : _downsample) {
        ofs << d.second[0] << " " << d.second[1] << " " << d.second[2] << std::endl;
    }

    ofs.close();
    std::cout << "save " << _downsample.size() << " points to " << file_name << std::endl;
    return true;
}