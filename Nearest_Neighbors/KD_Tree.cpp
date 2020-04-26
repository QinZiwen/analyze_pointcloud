#include "Nearest_Neighbors/KD_Tree.h"

namespace AAPCD {

void KDTree::input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix) {
    _input_matrix = input_matrix;
}

bool KDTree::create_kd_tree(int leaf_size) {
    std::vector<int> value_indices;
    for (int i = 0; i < _input_matrix.cols(); ++i) {
        value_indices.emplace_back(i);
        if (_verbose) {
            std::cout << "input_data: " << _input_matrix.col(i).transpose() << std::endl;
        }
    }
    return create_kd_tree_recursive(_root, _input_matrix, value_indices, 0, leaf_size);
}

bool KDTree::create_kd_tree_recursive(
    std::shared_ptr<KDTreeNode>& root,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& values,
    const std::vector<int> value_indices,
    int axis,
    int leaf_size) {
    
    if (_verbose) {
        std::cout << "axis: " << axis << std::endl;
    }
    if (root == nullptr) {
        root.reset(new KDTreeNode(axis));
    }

    if (_verbose) {
        std::cout << "value_indices.size: " << value_indices.size() << std::endl;
        std::cout << "value_indices: ";
        for (size_t i = 0; i < value_indices.size(); ++i) {
            std::cout << value_indices[i] << " ";
        }
        std::cout << std::endl;
    }

    if (value_indices.size() > leaf_size) {
        // get median
        double median_key;
        std::vector<int> left_value_indices, right_value_indices, median_value_indices;

        get_median_key_and_left_right_value_indices_maxmin(
            values, value_indices, axis, median_key, left_value_indices, right_value_indices, median_value_indices);

        if (_verbose) {
            std::cout << "median_key: " << median_key << std::endl;
            std::cout << "left_value_indices.size: " << left_value_indices.size() << std::endl;
            std::cout << "right_value_indices.size: " << right_value_indices.size() << std::endl;
        }

        int next_axis = get_next_axis(axis, values.rows());
        root->is_leaf = false;
        root->key = median_key;
        root->value_indices = median_value_indices;
        create_kd_tree_recursive(root->left, values, left_value_indices, next_axis, leaf_size);
        create_kd_tree_recursive(root->right, values, right_value_indices, next_axis, leaf_size);
    } else {
        root->is_leaf = true;
        root->value_indices = value_indices;

        // print value_indices
        if (_verbose) {
            std::cout << "leaf value_indices: ";
            for (size_t i = 0; i < value_indices.size(); ++i) {
                std::cout << value_indices[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    return true;
}

bool KDTree::get_median_key_and_left_right_value_indices_maxmin(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& values, 
    const std::vector<int> value_indices, 
    int axis,
    double& median_key,
    std::vector<int>& left_value_indices,
    std::vector<int>& right_value_indices,
    std::vector<int>& median_value_indices) {
    double max = 0, min = 0;

    Eigen::VectorXd tmp_values(value_indices.size());
    for (size_t i = 0; i < value_indices.size(); ++i) {
        tmp_values(i) = values(axis, value_indices[i]);
    }

    max = tmp_values.maxCoeff();
    min = tmp_values.minCoeff();
    if (_verbose) {
        std::cout << "min: " << min << ", max: " <<  max << std::endl;
    }

    if (max > min) {
        median_key = min + (max - min) / 2;
        for (size_t i = 0; i < value_indices.size(); ++i) {
            if (tmp_values(i) == median_key) {
                median_value_indices.emplace_back(value_indices[i]);
            } else if (tmp_values(i) < median_key) {
                left_value_indices.emplace_back(value_indices[i]);
            } else {
                right_value_indices.emplace_back(value_indices[i]);
            }
        }
    } else {
        median_key = min;
        int median_idx = floor(value_indices.size() / 2);
        if (tmp_values(median_key) == median_key) {
            median_value_indices = std::vector<int>(value_indices.begin(), value_indices.begin() + 1);
        }
        left_value_indices = std::vector<int>(value_indices.begin() + 1, value_indices.begin() + median_idx + 1);
        right_value_indices = std::vector<int>(value_indices.begin() + median_key + 1, value_indices.end());
    }
    return true;
}

int KDTree::get_next_axis(int axis, int dim) {
    if (axis >= dim - 1) {
        return 0;
    } else {
        return axis + 1;
    }
}

std::shared_ptr<KDTreeNode> KDTree::get_root() {
    return _root;
}

}  // namespace AAPCD