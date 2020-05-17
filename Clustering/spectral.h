#pragma once

#include "utils/utils.hpp"
#include "Clustering/KMeans.h"
#include "Nearest_Neighbors/nearest_neighbors.h"

#include <cmath>
#include <limits>

namespace AAPCD {

class Spectral {
public:
    enum class ADJACENCY_METHOD {
        NEAREST_NEIGHBOR,
        K_NEIGHBORHOOD_GRAPH,
        FULL_CONNECT
    };

    enum class NORMALIZED_LAPLACIAN {
        NONE,
        SYM,   // L = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
        RW     // D^{-1} L = I - D^{-1}W
    };

    void input(const Eigen::MatrixXd& input_matrix);
    bool compute(int k = 0,
                 int max_step = 100,
                 double min_update_size = 0.01,
                 ADJACENCY_METHOD adjacency = ADJACENCY_METHOD::NEAREST_NEIGHBOR,
                 NORMALIZED_LAPLACIAN normalized_laplacian = NORMALIZED_LAPLACIAN::RW);

    // 将聚类的结果保存到文件，每一行为一个类的所有数据
    bool save_cluster_data_to_file(const std::string& file_name);
    void print_clusters();

private:
    bool build_adjacency_matrix(ADJACENCY_METHOD adjacency = ADJACENCY_METHOD::FULL_CONNECT);
    bool build_adjacency_matrix_full_connect();
    bool build_adjacency_matrix_nearest_neighbor();

    bool build_Laplacian_matrix(NORMALIZED_LAPLACIAN normalized_laplacian = NORMALIZED_LAPLACIAN::NONE);
    bool build_Laplacian_matrix_none();
    bool build_Laplacian_matrix_RW();

    // V \in R^{n*k}, every column is smallest k eigenvectors of Laplacian matrix
    bool build_V_matrix(int k = 0);
    
private:
    Eigen::MatrixXd _data;
    Eigen::MatrixXd _W;
    Eigen::MatrixXd _L;
    Eigen::MatrixXd _V;

    int _num_cluster;
    std::vector<Cluster> _clusters;
};

}  // namespace AAPCD