#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "utils/utils.hpp"

class VoxelGridDownsampling {
public:
    enum class SelectPointsMethod {
        CENTROID,
        RANDOM
    };

    bool set_data(const std::string& pts_file);
    bool set_data(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& data);
    bool downsampling(double grid_size, SelectPointsMethod select_pts_method = SelectPointsMethod::CENTROID);
    bool save_to_file(const std::string& file_name);

    const std::vector<std::pair<int, Eigen::Vector3d>>& get_downsampel_result() {
        return _downsample;
    }

private:
    bool select_centroid(std::vector<std::pair<int, Eigen::Vector3d>>& downsample);
    bool select_random(std::vector<std::pair<int, Eigen::Vector3d>>& downsample);

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _pcd;
    std::vector<std::pair<int, Eigen::Vector3d>> _downsample;
};