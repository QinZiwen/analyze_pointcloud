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
#include "Nearest_Neighbors/nearest_neighbors.h"
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

        // ------- 之间的变量状态互斥 ---------
        _save_octree_with_color = false;   // 点云放到octree中后，每个格子里的点，随机染色

        // 保存每个点的法向量，以及估计的地面法向量，将地面法向量绑定到[0,0,0],[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]
        // _save_ground_normal 和 _save_octree_with_color 不能同时设置为true
        _save_ground_normal = false;

        // 是否保存根据地面法向量过滤出来的点
        _save_collect_inlier_using_plane_direction = false;
        _save_refine_ground_point = true;
        // ------- 之间的变量状态互斥 ---------

        is_downsample = true;  // 是否下采样点云
        vgd_grid_size = 0.5;      // 下采样点云时，grid的大小，单位为点云的原始点云

        // 用octree分割点云，然后在每个格子里估计平面的法向量。下面时octree的两个参数。
        octree_leaf_size = 30;
        octree_min_length = 10;

        _knn_radius = 1.2;

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
    bool collect_inlier_using_plane_direction();
    bool refine_ground_point();

private:
    Eigen::MatrixXd _input_data;
    Eigen::MatrixXd _data;
    Eigen::VectorXd _ground_param;

    bool _save_octree_to_file;
    bool _save_octree_with_color;
    bool _save_ground_normal;
    std::ofstream _save_octree_ofs;


    bool is_downsample;
    double vgd_grid_size;
    double octree_leaf_size;
    double octree_min_length;

    Octree octree;

    std::vector<int> _inlier_idx;
    bool _save_collect_inlier_using_plane_direction;
    double _knn_radius;
    
    bool _save_refine_ground_point;
};
}  // namespace AAPCD