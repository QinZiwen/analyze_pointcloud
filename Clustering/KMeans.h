#pragma once

#include <memory>
#include <vector>
#include <limits>  

#include <Eigen/Dense>
#include <Eigen/Core>

#include "utils/utils.hpp"

namespace AAPCD {

typedef struct Cluster {
    unsigned int id;
    Eigen::VectorXd center;
    std::vector<int> data_index;
    int data_index_size;
    double delta;

    Cluster(unsigned int id_)
    : id(id_) {
        delta = std::numeric_limits<double>::max();
        data_index_size = 0;
    }
} Cluster;

class KMeans {
public:
    // column major
    void input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix);
    bool compute(int k, int max_step = 100, double min_update_size = 0.01);
    std::vector<Cluster> get_clusters();
    
    // 将聚类的结果保存到文件，每一行为一个类的所有数据
    bool save_cluster_data_to_file(const std::string& file_name);
    void print_clusters();

private:
    bool init_cluster(int k);
    bool E_step();
    bool M_step();
    double get_update_size();

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _data;
    std::vector<Cluster> _clusters;
};

}  // namespace AAPCD