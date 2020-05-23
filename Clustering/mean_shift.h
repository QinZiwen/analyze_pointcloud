#pragma once

#include <memory>
#include <vector>
#include <limits>  

#include <Eigen/Dense>
#include <Eigen/Core>

#include "utils/utils.hpp"

namespace AAPCD {

typedef struct MeanShiftCluster {
    Eigen::VectorXd center;
    std::vector<Eigen::VectorXd> shifted_data;
    std::vector<int> data_index;
} MeanShiftCluster;

class MeanShift {
public:
    enum class KernalType {
        GAUSSIAN
    };

    enum class ClusterMethod {
        AUTO,
        KMEANS
    };

    // column major
    void input(const Eigen::MatrixXd& input_matrix);

    bool compute(double kernel_bandwidth, double epsilon = 0.001, ClusterMethod cluster_method = ClusterMethod::AUTO, int kmeans_k = 3);
    std::vector<MeanShiftCluster> get_clusters();
    
    // 将聚类的结果保存到文件，每一行为一个类的所有数据
    bool save_cluster_data_to_file(const std::string& file_name);
    void print_clusters();

private:
    bool calc_shifted_point(
        const Eigen::VectorXd& data, 
        double kernel_bandwidth,
        Eigen::VectorXd& shifted_point);

    bool kernel(double distance,
        double kernel_bandwidth,
        double& res,
        KernalType kernal_type = KernalType::GAUSSIAN);
    bool gaussian_kernel(double distance, double kernel_bandwidth, double& res);

private:
    Eigen::MatrixXd _data;
    Eigen::MatrixXd _shifted;

    int _kmeans_k;
    ClusterMethod _cluster_method;
    std::vector<MeanShiftCluster> _clusters;
};

}  // namespace AAPCD