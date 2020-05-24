#include "Clustering/spectral.h"

namespace AAPCD {

void Spectral::input(const Eigen::MatrixXd& input_matrix) {
    _data = input_matrix;
}
bool Spectral::compute(int k, int max_step,
    double min_update_size,
    ADJACENCY_METHOD adjacency,
    NORMALIZED_LAPLACIAN normalized_laplacian) {
    // build adjacency matrix
    if (!build_adjacency_matrix(adjacency)) {
        std::cerr << "run build_adjacency_matrix failure" << std::endl;
        return false;
    } else {
        std::cout << "build_adjacency_matrix success" << std::endl;
    }

    // compute Laplacian L
    if (!build_Laplacian_matrix(normalized_laplacian)) {
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
        std::cout << "build_adjacency_matrix: FULL_CONNECT" << std::endl;
        return build_adjacency_matrix_full_connect();
    } else if (adjacency == ADJACENCY_METHOD::NEAREST_NEIGHBOR) {
        std::cout << "build_adjacency_matrix: NEAREST_NEIGHBOR" << std::endl;
        return build_adjacency_matrix_nearest_neighbor();
    } else if (adjacency == ADJACENCY_METHOD::K_NEIGHBORHOOD_GRAPH) {
        std::cout << "build_adjacency_matrix: K_NEIGHBORHOOD_GRAPH" << std::endl;
    } else {
        std::cerr << "adjacency method invalid" << std::endl;
        return false;
    }
    return true;
}

bool Spectral::build_adjacency_matrix_full_connect() {
    _W.resize(_data.cols(), _data.cols());

    double max_weight = 0.0;
    for (size_t i = 0; i < _data.cols(); ++i) {
        const Eigen::VectorXd d1 = _data.col(i);
        for (size_t j = 0; j < _data.cols(); ++j) {
            if (i == j) {
                continue;
            }

            const Eigen::VectorXd d2 = _data.col(j);
            if ((d1 - d2).norm() > max_weight) {
                max_weight = (d1 - d2).norm();
            }
        }
    }

    for (size_t i = 0; i < _data.cols(); ++i) {
        const Eigen::VectorXd d1 = _data.col(i);
        for (size_t j = 0; j < _data.cols(); ++j) {
            if (i == j) {
                _W(i, j) = 0;
                continue;
            }
            const Eigen::VectorXd d2 = _data.col(j);
            _W(i, j) = max_weight - (d1 - d2).norm();
        }
    }
    return true;
}

bool Spectral::build_adjacency_matrix_nearest_neighbor() {
    int leaf_size = int(_data.cols() > 10 ? _data.cols() * 0.1 : 2);
    int knn_size = int(_data.cols() > 10 ? _data.cols() * 0.25 : 4);

    double max_weight = 0.0;
    for (size_t i = 0; i < _data.cols(); ++i) {
        const Eigen::VectorXd d1 = _data.col(i);
        for (size_t j = 0; j < _data.cols(); ++j) {
            const Eigen::VectorXd d2 = _data.col(j);
            if ((d1 - d2).norm() > max_weight) {
                max_weight = (d1 - d2).norm();
            }
        }
    }
    std::cout << "max_weight: " << max_weight << std::endl;
    std::cout << "leaf_size: " << leaf_size << std::endl;
    std::cout << "knn_size: " << knn_size << std::endl;

    _W.resize(_data.cols(), _data.cols());

    KDTreeAVLNearestNeighbors nn;
    nn.set_data(_data, leaf_size);

    for (size_t i = 0; i < _data.cols(); ++i) {
        const Eigen::VectorXd d1 = _data.col(i);
        KNNResultNumber knn_result(knn_size);
        nn.KNN_search_number(d1, knn_result);
        std::vector<DistanceValue> dv = knn_result.get_distance_value();
        for (size_t id = 0; id < dv.size(); ++id) {
            if (i == dv[id].value) {
                _W(i, dv[id].value) = 0;
                continue;
            }

            const Eigen::VectorXd d2 = _data.col(dv[id].value);
            _W(i, dv[id].value) = max_weight - (d1 - d2).norm();
        }
    }

    // std::cout << "weight matrix:\n" << _W << std::endl;
    // std::ofstream wofs("./weight.txt");
    // for (size_t i = 0; i < _data.cols(); ++i) {
    //     for (size_t j = 0; j < _data.cols(); ++j) {
    //         wofs << _W(i, j) << " ";
    //     }
    //     wofs << std::endl;
    // }
    // wofs.close();
    return true;
}

bool Spectral::build_Laplacian_matrix(NORMALIZED_LAPLACIAN normalized_laplacian) {
    if (normalized_laplacian == NORMALIZED_LAPLACIAN::NONE) {
        std::cout << "build_Laplacian_matrix: NONE" << std::endl;
        return build_Laplacian_matrix_none();
    } else if (normalized_laplacian == NORMALIZED_LAPLACIAN::SYM) {
        std::cout << "build_Laplacian_matrix: SYM" << std::endl;
    } else if (normalized_laplacian == NORMALIZED_LAPLACIAN::RW) {
        std::cout << "build_Laplacian_matrix: RW" << std::endl;
        return build_Laplacian_matrix_RW();
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

bool Spectral::build_Laplacian_matrix_RW() {
    // Degree of every point
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(_data.cols(), _data.cols());
    Eigen::VectorXd d = _W.rowwise().sum();
    for (size_t i = 0; i < _data.cols(); ++i) {
        D(i, i) = d(i);
    }

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(_data.cols(), _data.cols());
    _L = I - D.inverse() * _W;
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
    return true;
}

void Spectral::print_clusters() {
    std::cout << "===== spectral clusters =====" << std::endl;
    for (const auto& c : _clusters) {
        std::cout << "cluster " << c.id << ", " << c.center.transpose() << ", " << c.data_index_size << std::endl;
        std::cout << "data index: ";
        for (int i = 0; i < c.data_index_size; ++i) {
            std::cout << c.data_index[i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "====================" << std::endl;
}

}  // namespace AAPCD