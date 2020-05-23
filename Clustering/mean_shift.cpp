#include "Clustering/mean_shift.h"

#include "Clustering/KMeans.h"
#include <limits>

namespace AAPCD {

void MeanShift::input(const Eigen::MatrixXd& input_matrix) {
    _data = input_matrix;
}

bool MeanShift::compute(double kernel_bandwidth, double epsilon, ClusterMethod cluster_method, int kmeans_k) {
    _cluster_method = cluster_method;
    _kmeans_k = kmeans_k;

    _shifted = _data;
    double max_shift = 0.;
    std::vector<double> stop_moving(_data.cols(), std::numeric_limits<double>::max());

    do {
        max_shift = 0;
        for (size_t i = 0; i < _shifted.cols(); ++i) {
            if (stop_moving[i] > epsilon) {
                // calc shifted point
                Eigen::VectorXd shifted_point;
                if (!calc_shifted_point(_shifted.col(i), kernel_bandwidth, shifted_point)) {
                    std::cerr << "run calc_shifted_point failure" << std::endl;
                    return false;
                }
                // std::cout << "_shifted: " << _shifted.col(i).transpose() << std::endl;
                // std::cout << "shifted_point: " << shifted_point.transpose() << std::endl;

                double shift_dis = (_shifted.col(i) - shifted_point).norm();
                // std::cout << "shift_dis: " << shift_dis << std::endl;

                if (shift_dis > max_shift) {
                    max_shift = shift_dis;
                }
                if (shift_dis < stop_moving[i]) {
                    stop_moving[i] = shift_dis;
                }
                _shifted.col(i) = shifted_point;
            }
        }
    } while(max_shift > epsilon);

    // std::cout << "data: \n" << _data.leftCols(5) << std::endl;
    // std::cout << "_shifted: \n" << _shifted.leftCols(5) << std::endl;

    return true;
}

bool MeanShift::calc_shifted_point(
    const Eigen::VectorXd& data, 
    double kernel_bandwidth,
    Eigen::VectorXd& shifted_point) {
    shifted_point = Eigen::VectorXd::Zero(data.size());
    double total_weight = 0.;
    for (size_t i = 0; i < _data.cols(); ++i) {
        Eigen::VectorXd tmp = _data.col(i);
        double dis = (tmp - data).norm();

        double weight = 0;
        if (!kernel(dis, kernel_bandwidth, weight)) {
            std::cerr << "run kernel failure" << std::endl;
            return false;
        }
        shifted_point += weight * tmp;
        total_weight += weight;
    }

    shifted_point /= total_weight;
    return true;
}

bool MeanShift::gaussian_kernel(double distance, double kernel_bandwidth, double& res){
    res = exp(-1.0/2.0 * (distance*distance) / (kernel_bandwidth*kernel_bandwidth));
    return true;
}

bool MeanShift::kernel(double distance, double kernel_bandwidth, double& res, KernalType kernal_type) {
    switch (kernal_type)
    {
    case KernalType::GAUSSIAN:
        return gaussian_kernel(distance, kernel_bandwidth, res);
    default:
        std::cerr << "Invalid kernal type" << std::endl;
        return false;
    }
}

std::vector<MeanShiftCluster> MeanShift::get_clusters() {
    _clusters.clear();
    switch (_cluster_method)
    {
    case ClusterMethod::AUTO:
        {
            // 寻找最小的shifted point之间的距离
            double min_dis_cluster = 0;
            for (size_t i = 0; i < _shifted.cols() - 1; ++i) {
                for (size_t j = i + 1; j < _shifted.cols(); ++j) {
                    double dis = (_shifted.col(i) - _shifted.col(j)).norm();
                    if (dis > min_dis_cluster) {
                        min_dis_cluster = dis;
                    }
                }
            }
            min_dis_cluster /= 5;
            std::cout << "min_dis_cluster: " << min_dis_cluster << std::endl;
            

            for (size_t i = 0; i < _shifted.cols(); ++i) {
                // double min_dis_cluster = 0.1;
                // 寻找当点距离最近的类中心
                size_t c_center = 0;
                for (; c_center < _clusters.size(); ++c_center) {
                    double dis = (_shifted.col(i) - _clusters[c_center].center).norm();
                    if (dis < min_dis_cluster) {
                        break;
                    }
                }

                if (c_center == _clusters.size()) {   // 没有找到
                    MeanShiftCluster c;
                    c.center = _shifted.col(i);
                    _clusters.emplace_back(c);
                }
                _clusters[c_center].shifted_data.emplace_back(_shifted.col(i));
                _clusters[c_center].data_index.emplace_back(i);
            }
        }
        break;
    case ClusterMethod::KMEANS:
        {
            // 用k-means去找类中心
            KMeans kmean;
            kmean.input(_shifted);
            if (!kmean.compute(_kmeans_k)){
                std::cerr << "k-mean compute failure" << std::endl;
            }
            std::vector<Cluster> kmean_cluster = kmean.get_clusters();
            for (auto cs : kmean_cluster) {
                MeanShiftCluster c;
                c.center = cs.center;
                _clusters.emplace_back(c);
            }

            std::cout << "cluster size: " << _clusters.size() << std::endl;
            for (size_t i = 0; i < _data.cols(); ++i) {
                double min_dis_cluster = std::numeric_limits<double>::max();
                size_t c_center = -1;
                for (size_t c = 0; c < _clusters.size(); ++c) {
                    double dis = (_data.col(i) - _clusters[c].center).norm();
                    if (dis < min_dis_cluster) {
                        c_center = c;
                        min_dis_cluster = dis;
                    }
                }

                if (c_center == -1 || c_center == _clusters.size()) {   // 没有找到
                    std::cerr << "Can not find a cluster center" << std::endl;
                    break;
                }
                _clusters[c_center].shifted_data.emplace_back(_shifted.col(i));
                _clusters[c_center].data_index.emplace_back(i);
            }
        }
        break;
    default:
        std::cerr << "Invalid ClusterMethod" << std::endl;
        break;
    }
    
    return _clusters;
}

void MeanShift::print_clusters() {
    std::cout << "=====  MeanShiftCluster =====" << std::endl;
    for (const auto& c : _clusters) {
        std::cout << "center: " << c.center.transpose() << ", number of data: " << c.data_index.size() << std::endl;
    }
    std::cout << "=============================" << std::endl;
}

bool MeanShift::save_cluster_data_to_file(const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    // cluster
    get_clusters();
    for (const auto& c : _clusters) {
        for (size_t i = 0; i < c.data_index.size(); ++i) {
            for (size_t row = 0; row < _data.rows(); ++row) {
                ofs << _data(row, c.data_index[i]) << " ";
                // ofs << c.shifted_data[i][row] << " ";
            }
        }
        ofs << std::endl;
    }

    ofs.close();
    std::cout << "save result to " << file_name << std::endl;
    return true;
}

}  // namespace AAPCD