#include "Clustering/GMM.h"

#include <cstdlib>
#include <ctime>
#include <set>

namespace AAPCD {

void GMM::input(const Eigen::MatrixXd& input_matrix) {
    _data = input_matrix;
}

bool GMM::compute(int k, int max_step, double min_update_size) {
    // 初始化k个高斯
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

bool GMM::init_cluster(int k) {
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

        GMMCLuster cluster(i, 1.0 / k);
        cluster.mu = _data.col(idx);
        cluster.sigma = Eigen::MatrixXd::Identity(_data.rows(), _data.rows());
        _clusters.emplace_back(cluster);

        idx_removed.insert(idx);
        ++i;
    }

    _Z_nk = Eigen::MatrixXd::Zero(_data.cols(), k);
    print_clusters();

    return true;
}

bool GMM::E_step() {
    for (size_t n = 0; n < _data.cols(); ++n) {
        // total probability 
        double total_prob = 0;
        for (size_t k = 0; k < _Z_nk.cols(); ++k) {
            total_prob += _clusters[k].pi * _clusters[k].probability(_data.col(n));
        }

        for (size_t k = 0; k < _Z_nk.cols(); ++k) {
            double prob = _clusters[k].pi * _clusters[k].probability(_data.col(n));
            _Z_nk(n, k) = prob / total_prob;
        }
    }
    return true;
}

bool GMM::M_step() {
    Eigen::VectorXd Z_nk_sum = _Z_nk.colwise().sum();
    for (size_t k = 0; k < _Z_nk.cols(); ++k) {
        double N_k = Z_nk_sum[k];

        // update mu
        Eigen::VectorXd new_mu_ = (_Z_nk.col(k).transpose() * _data.transpose()) / N_k;
        Eigen::VectorXd new_mu = new_mu_.transpose();

        // update sigma
        Eigen::MatrixXd new_sigma = Eigen::MatrixXd::Zero(_data.rows(), _data.rows());
        for (size_t n = 0; n < _data.cols(); ++n) {
            new_sigma += _Z_nk(n, k) * (_data.col(n) - new_mu) * (_data.col(n) - new_mu).transpose();
        }
        new_sigma /= N_k;

        // update pi
        double new_pi = N_k / _data.cols();

        if (new_sigma.determinant() > 1e-8) {
            _clusters[k].delta = (_clusters[k].mu - new_mu).norm();
            _clusters[k].mu = new_mu;
            _clusters[k].sigma = new_sigma;
            _clusters[k].pi = new_pi;
        } else {
            srand((int)time(0));
            int idx = rand() % _data.cols();
            _clusters[k].mu = _data.col(idx);
            _clusters[k].sigma = Eigen::MatrixXd::Identity(_data.rows(), _data.rows());
        }
    }
    return true;
}

void GMM::print_clusters() {
    std::cout << "===== GMMCluster =====" << std::endl;
    for (const auto& c : _clusters) {
        std::cout << "GMMCluster " << c.id << ", mu: " << c.mu.transpose()
            << ", pi: " << c.pi << ", delta: " << c.delta
            << ", sigma: \n" << c.sigma << std::endl;
    }
    std::cout << "====================" << std::endl;
}

double GMM::get_update_size() {
    double up_size = 0;
    for (const auto& c : _clusters) {
        if (c.delta > up_size) {
            up_size = c.delta;
        }
    }
    return up_size;
}

void GMM::print_Z_nk() {
    std::cout << "===== GMM Znk =====" << std::endl;
    for (size_t n = 0; n < _data.cols(); ++n) {
        std::cout << _data.col(n).transpose() << " ";
        for (size_t k = 0; k < _Z_nk.cols(); ++k) {
            std::cout << _Z_nk(n, k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "===================" << std::endl;
}

bool GMM::save_cluster_data_to_file(const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    for (size_t n = 0; n < _data.cols(); ++n) {
        ofs << _data.col(n).transpose();

        for (size_t k = 0; k < _Z_nk.cols(); ++k) {
            ofs << " " << _Z_nk(n, k);
        }
        ofs << std::endl;
    }

    ofs.close();
    std::cout << "save result to " << file_name << std::endl;
    return true;
}

}  // namespace AAPCD