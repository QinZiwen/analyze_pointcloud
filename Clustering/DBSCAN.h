#pragma once

#include <memory>
#include <vector>
#include <limits>  

#include <Eigen/Dense>
#include <Eigen/Core>

#include "utils/utils.hpp"
#include "Nearest_Neighbors/nearest_neighbors.h"

namespace AAPCD {

typedef struct DBSCANCluster {
    std::vector<size_t> core_point;
    std::vector<size_t> neighborhood;
} DBSCANCluster;

class DBSCAN {
public:
    void input(const Eigen::MatrixXd& input_data, int leaf_size = 5);
    bool compute(double radius, int min_points);

    std::vector<DBSCANCluster> get_clusters();
    // 将聚类的结果保存到文件，每一行为一个类的所有数据
    bool save_cluster_data_to_file(const std::string& file_name);
    void print();

private:
    bool find_knn(const Eigen::VectorXd& data, double radius, std::vector<size_t>& neighbors);

private:
    Eigen::MatrixXd _data;
    std::vector<DBSCANCluster> _clusters;

    KDTreeAVLNearestNeighbors _kdtree_nn;
};

}  // namespace AAPCD