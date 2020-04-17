#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "utils/utils.hpp"

class PCA {
public:
    enum class eigen_vector_order : uint8_t{
        ASCENDING, DESCENDING
    };

    void input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix);
    bool compute(eigen_vector_order order = eigen_vector_order::DESCENDING);

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& get_input_matrix();
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& get_eigen_values();
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& get_eigen_vector();

private:
    void sort_eigen_vector(eigen_vector_order order = eigen_vector_order::DESCENDING);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _input_matrix;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _centered_matrix;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _covariance_matrix;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _eigen_values;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _eigen_vectors;
};  // class PCA