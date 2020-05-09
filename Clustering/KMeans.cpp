#include "Clustering/KMeans.h"

#include <cstdlib>
#include <ctime>
#include <set>

namespace AAPCD {

void KMeans::input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix) {
    _data = input_matrix;
}

bool KMeans::compute(int k, int max_step, double min_update_size) {
    // 初始化k个center
    if (!init_cluster(k)) {
        std::cerr << "initial cluster failure" << std::endl;
        return false;
    }
    int i = 0;
    for (i = 0; i < max_step; ++i) {
        std::cout << "iterations: " << i << std::endl;

        // E-step
        if (!E_step()) {
            std::cerr << "E_step failure" << std::endl;
            return false;
        }

        // M-step
        if (!M_step()) {
            std::cerr << " M_step failure" << std::endl;
            return false;
        }

        // 根据更新两判断是否要退出
        double update_size = get_update_size();
        std::cout << "update_size: " << update_size << std::endl;
        if (update_size < min_update_size) {
            std::cout << "update_size < min_update_size: " << update_size << " : " << min_update_size << std::endl;
            break;
        }
    }
    if (i >= max_step) {
        std::cout << "reach max_step: " << max_step << std::endl;
    }
    return true;
}

bool KMeans::init_cluster(int k) {
    size_t data_size = _data.cols();

    if (k > data_size) {
        std::cerr << "k > data_size: " << k << " : " << data_size << std::endl;
        return false;
    }

    srand((int)time(0));
    std::set<int> idx_removed;
    for (int i = 0; i < k;) {
        int idx = rand() % data_size;
        if (idx_removed.count(idx) != 0) {
            continue;
        }

        Cluster cluster(i);
        cluster.center = _data.col(idx);
        cluster.data_index.resize(data_size);
        cluster.data_index_size = 0;
        _clusters.emplace_back(cluster);
        idx_removed.insert(idx);
        ++i;
    }

    print_clusters();

    return true;
}

bool KMeans::E_step() {
    if (_clusters.size() <= 0) {
        std::cerr << "clusters size is zero" << std::endl;
        return false;
    }

    // clear cluster
    for (Cluster& c : _clusters) {
        c.data_index_size = 0;
    }

    for (int i = 0; i < _data.cols(); ++i) {
        const Eigen::VectorXd feature = _data.col(i);
        Cluster* cluster_ptr = nullptr;
        double distance = std::numeric_limits<double>::max();
        for (Cluster& c : _clusters) {
            Eigen::VectorXd dis_vector = feature - c.center;
            double tmp_dis = dis_vector.norm();
            if (tmp_dis < distance) {
                distance = tmp_dis;
                cluster_ptr = &c;
            }
        }

        if (cluster_ptr == nullptr) {
            std::cerr << "No " << i << " featuren can not fine cluster center" << std::endl;
            return false;
        }

        // std::cout << "E-step: cluster " << cluster_ptr->id << ", " << cluster_ptr->center.transpose() << ", " << i << std::endl;
        cluster_ptr->data_index[cluster_ptr->data_index_size++] = i;
    }

    // print_clusters();

    return true;
}

bool KMeans::M_step() {
    if (_clusters.size() <= 0) {
        std::cerr << "clusters size is zero" << std::endl;
        return false;
    }

    for (Cluster& c : _clusters) {
        if (c.data_index_size <= 0) {
            continue;
        }

        Eigen::VectorXd feature = Eigen::VectorXd::Zero(_data.rows());
        for (int i = 0; i < c.data_index_size; ++i) {
            feature += _data.col(c.data_index[i]);
        }
        Eigen::VectorXd new_center = feature / c.data_index_size;
        c.delta = (new_center - c.center).norm();
        c.center = new_center;
    }
    // print_clusters();

    return true;
}

double KMeans::get_update_size() {
    double up_size = 0;
    for (const auto& c : _clusters) {
        if (c.delta > up_size) {
            up_size = c.delta;
        }
    }
    return up_size;
}

std::vector<Cluster> KMeans::get_clusters() {
    return _clusters;
}

void KMeans::print_clusters() {
    std::cout << "===== clusters =====" << std::endl;
    for (const auto& c : _clusters) {
        std::cout << "cluster " << c.id << ", " << c.center.transpose() << ", " << c.data_index_size << std::endl;
    }
    std::cout << "====================" << std::endl;
}

bool KMeans::save_cluster_data_to_file(const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    for (const auto& c : _clusters) {
        for (size_t i = 0; i < c.data_index_size; ++i) {
            ofs << _data(0, c.data_index[i]) << " " << _data(1, c.data_index[i]) << " ";
        }
        ofs << std::endl;
    }

    ofs.close();
    std::cout << "save result to " << file_name << std::endl;
    return true;
}

}  // namespace AAPCD