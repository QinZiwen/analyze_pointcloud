#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "KD_tree/KD_tree.hpp"

class NormalEstimator {
public:
    bool set_data(const std::string& pts_file);
    bool compute(double threshold, const std::string& output_file_name = std::string());

private:
    pointVec _points;
    KDTree _kd_tree;
};