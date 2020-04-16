#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "utils/utils.hpp"

class VoxelGridDownsampling {
public:
    bool set_data(const std::string& pts_file);
    bool downsampling(int grid_size);
    bool save_to_file(const std::string& file_name);

private:
    bool select_centroid(std::vector<std::pair<int, Eigen::Vector3d>>& downsample);
    bool select_random(std::vector<std::pair<int, Eigen::Vector3d>>& downsample);

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _pcd;
    std::vector<std::pair<int, Eigen::Vector3d>> _downsample;
};