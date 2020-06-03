#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <limits>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "utils/utils.hpp"
#include "Nearest_Neighbors/Octree.h"
#include "PCA_Downsample/pca.h"
#include "Clustering/KMeans.h"
#include "PCA_Downsample/voxel_grid_downsampling.h"

namespace AAPCD {
class FindGround {
public:
    enum class METHOD {
        FG_OCTREE
    };

    FindGround() {
        // _save_octree_to_file设置为true后，需要将_save_octree_with_color或者_save_ground_normal设置为true
        _save_octree_to_file = true;
        _save_octree_with_color = false;   // 点云放到octree中后，每个格子里的点，随机染色

        // 保存每个点的法向量，以及估计的地面法向量，将地面法向量绑定到[0,0,0],[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]
        // _save_ground_normal 和 _save_octree_with_color 不能同时设置为true
        _save_ground_normal = false;

        if (_save_octree_to_file) {
            _save_octree_ofs.open("octree_data_with_color.txt");
            if (!_save_octree_ofs.is_open()) {
                std::cerr << "Can not open octree_data_with_color.txt" << std::endl;
            }
            srand(time(0));
        }   
    }

    ~FindGround() {
        if (_save_octree_to_file && _save_octree_ofs.is_open()) {
            _save_octree_ofs.close();
        }
    }

    void input(const Eigen::MatrixXd& input_matrix);
    bool find(const METHOD& method = METHOD::FG_OCTREE);
    bool find_use_octree();

private:
    bool find_ground_direction(const std::shared_ptr<Octant>& octants);
    bool fitting_plane_RANSAC(const Eigen::MatrixXd& points, int min_set, int max_iter, Eigen::VectorXd& best_param);
    bool is_share_plane(const Eigen::MatrixXd& points);
    bool cluster_plane_param(const std::vector<Eigen::VectorXd>& plane_param_vec, Eigen::VectorXd& param);
    double distance_plane(const Eigen::VectorXd& point);

private:
    Eigen::MatrixXd _input_data;
    Eigen::MatrixXd _data;
    Eigen::VectorXd _ground_param;

    bool _save_octree_to_file;
    bool _save_octree_with_color;
    bool _save_ground_normal;
    std::ofstream _save_octree_ofs;
};
}  // namespace AAPCD