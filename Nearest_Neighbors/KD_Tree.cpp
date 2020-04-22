#include "Nearest_Neighbors/KD_Tree.h"

namespace AAPCD {

void KDTree::input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix) {
    _input_matrix = input_matrix;
}

bool KDTree::create_kd_tree_recursive(
    std::shared_ptr<KDTreeNode>& root,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& values,
    const std::vector<int> value_indices,
    int axis,
    int leaf_size) {
    if (root == nullptr) {
        root.reset(new KDTreeNode(axis));
    }

    if (value_indices.size() > leaf_size) {
        // get median
        root->is_leaf = false;
        double median_key;
        std::vector<int> left_value_indices, right_value_indices;

        get_median_key_and_left_right_value_indices_maxmin(values, value_indices, axis, median_key, left_value_indices, right_value_indices);

        int next_axis = get_next_axis(axis, values.rows());
        create_kd_tree_recursive(root->left, values, left_value_indices, next_axis, leaf_size);
        create_kd_tree_recursive(root->right, values, right_value_indices, next_axis, leaf_size);
    } else {
        root->is_leaf = true;
        root->value_indices = value_indices;
    }
    return true;
}

bool KDTree::get_median_key_and_left_right_value_indices_maxmin(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& values, 
    const std::vector<int> value_indices, 
    int axis,
    int& median_key,
    std::vector<int>& left_value_indices,
    std::vector<int>& right_value_indices) {
    double max = 0, min = 0;

    Eigen::VectorXd tmp_values(value_indices.size());
    if (value_indices.size() != values.row(axis).cols()) {
        for (size_t i = 0; i < value_indices.size(); ++i) {
            tmp_values(i) = values(axis, i);
        }
    } else {
        tmp_values = values.row(axis);
    }

    max = tmp_values.maxCoeff();
    min = tmp_values.minCoeff();
    if (max > min) {
        median_key = min + (max - min) / 2;
        for (size_t i = 0; i < value_indices.size(); ++i) {
            if (tmp_values(i) < median_key) {
                left_value_indices.emplace_back(value_indices[i]);
            } else {
                right_value_indices.emplace_back(value_indices[i]);
            }
        }
    } else {
        median_key = min;
        int median_idx = floor(value_indices.size() / 2);
        left_value_indices = std::vector<int>(value_indices.begin(), value_indices.begin() + median_idx - 1);
        right_value_indices = std::vector<int>(value_indices.begin() + median_key + 1, value_indices.end());
    }
    return true;
}

int KDTree::get_next_axis(int axis, int dim) {
    if (axis >= dim) {
        return 0;
    } else {
        return axis + 1;
    }
}

}  // namespace AAPCD