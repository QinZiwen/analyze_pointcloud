#pragma once

#include "utils/utils.hpp"
#include <cmath>
#include <limits>

namespace AAPCD {

typedef struct GMMCluster {
    int id;
    Eigen::VectorXd mu;
    Eigen::MatrixXd sigma;
    double pi;
    double delta;

    GMMCluster(int id_, double pi_)
    : id(id_), pi(pi_) {
        delta = std::numeric_limits<double>::max();
    }

    // 高斯
    double probability(const Eigen::VectorXd& x) {
        Eigen::VectorXd exp_index = (x - mu).transpose()
             * sigma.inverse() * (x - mu);
        // std::cout << "exp_index: " << exp_index.transpose() << std::endl;
        exp_index *= -1. / 2;
        // std::cout << "exp_index: " << exp_index.transpose() << std::endl;

        double coefficient = (1 / pow(2 * M_PI, mu.size() / 2)) * 
            (1 / pow(sigma.determinant(), 0.5));
        double result = coefficient * exp(exp_index[0]);
        return result;
    }
} GMMCLuster;

class GMM {
public:
    // 每一列为一个数据
    void input(const Eigen::MatrixXd& input_matrix);
    bool compute(int k, int max_step = 100, double min_update_size = 0.01);
    std::vector<GMMCLuster> get_clusters();

    bool save_cluster_data_to_file(const std::string& file_name);
    void print_clusters();
    void print_Z_nk();

private:
    bool init_cluster(int k);
    bool E_step();
    bool M_step();
    double get_update_size();

private:
    Eigen::MatrixXd _data;
    Eigen::MatrixXd _Z_nk;
    std::vector<GMMCLuster> _clusters;
};

}  // namespace AAPCD