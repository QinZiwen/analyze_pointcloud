#include "Clustering/spectral.h"

namespace AAPCD {

void Spectral::input(const Eigen::MatrixXd& input_matrix) {
    _data = input_matrix;
}
bool Spectral::compute(int k, int max_step, double min_update_size) {
    // build adjacency matrix
    if (!build_adjacency_matrix()) {
        std::cerr << "run build_adjacency_matrix failure" << std::endl;
        return false;
    } else {
        std::cout << "build_adjacency_matrix success" << std::endl;
    }

    // compute Laplacian L
    if (!build_Laplacian_matrix()) {
        std::cerr << "run build_Laplacian_matrix failure" << std::endl;
        return false;
    } else {
        std::cout << "build_Laplacian_matrix success" << std::endl;
    }

    // compute V matrix
    if (!build_V_matrix(k)) {
        std::cerr << "run build_V_matrix failure" << std::endl;
        return false;
    } else {
        std::cout << "build_V_matrix success" << std::endl;
    }

    // k-means
    KMeans kmeans;
    kmeans.input(_V.transpose());
    if (!kmeans.compute(_num_cluster, max_step, min_update_size)) {
        std::cerr << "kmeans compute failure" << std::endl;
        return false;
    } else {
        std::cout << "kmeans compute success" << std::endl;
    }
    kmeans.print_clusters();
    _clusters = kmeans.get_clusters();

    return true;
}

bool Spectral::build_adjacency_matrix(ADJACENCY_METHOD adjacency) {
    if (adjacency == ADJACENCY_METHOD::FULL_CONNECT) {
        return build_adjacency_matrix_full_connect();
    } else if (adjacency == ADJACENCY_METHOD::NEAREST_NEIGHBOR) {
        ;
    } else if (adjacency == ADJACENCY_METHOD::K_NEIGHBORHOOD_GRAPH) {
        ;
    } else {
        std::cerr << "adjacency method invalid" << std::endl;
        return false;
    }
    return true;
}

bool Spectral::build_adjacency_matrix_full_connect() {
    _W.resize(_data.cols(), _data.cols());
    for (size_t i = 0; i < _data.cols(); ++i) {
        const Eigen::VectorXd d1 = _data.col(i);
        for (size_t j = 0; j < _data.cols(); ++j) {
            const Eigen::VectorXd d2 = _data.col(j);
            _W(i, j) = (d1 - d2).norm();
        }
    }
    return true;
}

bool Spectral::build_Laplacian_matrix(NORMALIZED_LAPLACIAN normalized_laplacian) {
    if (normalized_laplacian == NORMALIZED_LAPLACIAN::NONE) {
        return build_Laplacian_matrix_none();
    } else if (normalized_laplacian == NORMALIZED_LAPLACIAN::SYM) {
        ;
    } else if (normalized_laplacian == NORMALIZED_LAPLACIAN::RW) {
        ;
    } else {
        std::cerr << "normalized laplacian invalid" << std::endl;
        return false;
    }
    return true;
}

bool Spectral::build_Laplacian_matrix_none() {
    // Degree of every point
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(_data.cols(), _data.cols());
    Eigen::VectorXd d = _W.rowwise().sum();
    for (size_t i = 0; i < _data.cols(); ++i) {
        D(i, i) = d(i);
    }
    _L = D - _W;
    return true;
}

bool Spectral::build_V_matrix(int k) {
    // eigen values and eigen vector
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> eigen_solver(_L);
    // sorted in increasing order
    Eigen::VectorXd L_eigen_values = eigen_solver.eigenvalues();
    Eigen::MatrixXd L_eigen_vectors = eigen_solver.eigenvectors();
    std::cout << "L_eigen_values: " << L_eigen_values.transpose() << std::endl;

    if (k != 0) {
        _num_cluster = k;
    } else {
        // fine eigen-gap
        if (L_eigen_values.size() < 3) {
            _num_cluster = L_eigen_values.size();
        } else {
            double thresh = 0;
            // init thresh
            for (int i = 0; i < L_eigen_values.size(); ++i) {
                thresh += L_eigen_values[i];
            }
            thresh /= L_eigen_values.size();

            for (int i = 0; i < L_eigen_values.size()-1; ++i) {
                if (abs(L_eigen_values[i+1] - L_eigen_values[i]) > thresh) {
                    _num_cluster = i;
                    break;
                } else {
                    thresh = abs(L_eigen_values[i+1] - L_eigen_values[i]);
                }
            }
        }
    }

    std::cout << "number of cluster: " << _num_cluster << std::endl;
    _V = L_eigen_vectors.leftCols(_num_cluster);

    return true;
}

bool Spectral::save_cluster_data_to_file(const std::string& file_name) {
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
    return false;
}

}  // namespace AAPCD