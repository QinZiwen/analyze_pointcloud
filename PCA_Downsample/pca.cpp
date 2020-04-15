#include "PCA_Downsample/pca.h"

#include <iostream>
#include <vector>

void PCA::input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix) {
    _input_matrix = input_matrix;
}

bool PCA::compute(eigen_vector_order order) {
    // Normalized by the center
    _centered_matrix = _input_matrix.colwise() - _input_matrix.rowwise().mean();
    // std::cout << _centered_matrix << std::endl;

    if (_input_matrix.cols() == 1) {
        _covariance_matrix = (_centered_matrix * _centered_matrix.adjoint());
    } else {
        _covariance_matrix = (_centered_matrix * _centered_matrix.adjoint()) / (_input_matrix.cols() - 1);
    }

    // eigen values and eigen vector
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> eigen_solver(_covariance_matrix);
    // sorted in increasing order
    _eigen_values = eigen_solver.eigenvalues();
    _eigen_vectors = eigen_solver.eigenvectors();

    sort_eigen_vector(order);
    return true;
}

const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& PCA::get_input_matrix() {
    return _input_matrix;
}
const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& PCA::get_eigen_values() {
    return _eigen_values;
}
const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& PCA::get_eigen_vector() {
    return _eigen_vectors;
}

void PCA::sort_eigen_vector(eigen_vector_order order) {
    std::vector<std::pair<int, int>> eigen_value_index_vector;
    for (int i = 0; i < _eigen_values.size(); ++i){
        eigen_value_index_vector.push_back(std::make_pair(_eigen_values(i, 0), i));
    }

    if (order == eigen_vector_order::ASCENDING)
        std::sort(std::begin(eigen_value_index_vector), std::end(eigen_value_index_vector), std::less<std::pair<int, int>>());
    else
        std::sort(std::begin(eigen_value_index_vector), std::end(eigen_value_index_vector), std::greater<std::pair<int, int>>());

    auto sorted_eigen_values = Eigen::Matrix<double, Eigen::Dynamic, 1>(_eigen_values.rows());
    auto sorted_eigen_vectors = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(_eigen_vectors.rows(), _eigen_vectors.cols());
    for (int i = 0; i < _eigen_values.size(); ++i) {
        sorted_eigen_values(i, 0) = _eigen_values(eigen_value_index_vector[i].second, 0);
        sorted_eigen_vectors.col(i) = _eigen_vectors.col(eigen_value_index_vector[i].second);
    }

    _eigen_values = sorted_eigen_values;
    _eigen_vectors = sorted_eigen_vectors;
}