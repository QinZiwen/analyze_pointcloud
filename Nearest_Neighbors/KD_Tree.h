#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "utils/utils.hpp"

namespace AAPCD {

typedef struct KDTreeNode {
    int axis;
    double key;
    bool is_leaf;

    std::shared_ptr<struct KDTreeNode> left;
    std::shared_ptr<struct KDTreeNode> right;
    std::vector<int> value_indices;

    KDTreeNode()
    : axis(-1), key(0), is_leaf(false) {}
    KDTreeNode(int axis_)
    : axis(axis_), key(0), is_leaf(false) {}
} KDTreeNode;

class KDTree {
public:
    KDTree(bool verbose = false) : _verbose(verbose) {}
    void input(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix);
    bool create_kd_tree(int leaf_size);

    std::shared_ptr<KDTreeNode> get_root();

private:
    bool create_kd_tree_recursive(
        std::shared_ptr<KDTreeNode>& root,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& values,
        const std::vector<int> value_indices,
        int axis,
        int leaf_size);
    
    bool get_median_key_and_left_right_value_indices_maxmin(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& values, 
        const std::vector<int> value_indices, 
        int axis,
        double& median_key,
        std::vector<int>& left_value_indices,
        std::vector<int>& right_value_indices,
        std::vector<int>& median_value_indices);
    
    int get_next_axis(int axis, int dim);

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _input_matrix;   // column is feature
    std::shared_ptr<KDTreeNode> _root;
    bool _verbose;
};

}  // namespace AAPCD
