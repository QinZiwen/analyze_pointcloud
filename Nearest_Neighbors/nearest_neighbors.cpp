#include "Nearest_Neighbors/nearest_neighbors.h"

bool AVLNearestNeighbors::set_data(const std::vector<double>& data) {
    _savl_tree.insert_from_vector(data);
    return true;
}

bool AVLNearestNeighbors::KNN_search_number(double key, KNNResultNumber& knn_result) {
    return KNN_search_number(_savl_tree.get_root(), key, knn_result);
}

bool AVLNearestNeighbors::KNN_search_number(const std::shared_ptr<SAVLNode>& savl_tree, double key, KNNResultNumber& knn_result) {
    if (savl_tree == nullptr) {
        return false;
    }

    // 如果knn_result的worst distance == 0，说明所有最近的value都找到了
    if (knn_result.worst_distance() == 0) {
        return true;
    }

    // 将value set到knn_result中
    knn_result.add_result(std::fabs(key - savl_tree->value), savl_tree->value);

    // 根据key去不同的子树查找
    if (key < savl_tree->value) {
        if (KNN_search_number(savl_tree->left, key, knn_result)) {
            return true;
        }
        if (std::fabs(key - savl_tree->value) < knn_result.worst_distance()) {
            return KNN_search_number(savl_tree->right, key, knn_result);
        } else {
            return false;
        }
    } else {
        if (KNN_search_number(savl_tree->right, key, knn_result)) {
            return true;
        }
        if (std::fabs(key - savl_tree->value) < knn_result.worst_distance()) {
            return KNN_search_number(savl_tree->left, key, knn_result);
        } else {
            return false;
        }
    }

    return true;
}

bool AVLNearestNeighbors::KNN_search_radius(double key, KNNResultRadius& knn_result) {
    return KNN_search_radius(_savl_tree.get_root(), key, knn_result);
}

bool AVLNearestNeighbors::KNN_search_radius(const std::shared_ptr<SAVLNode>& savl_tree, double key, KNNResultRadius& knn_result) {
    if (savl_tree == nullptr) {
        return false;
    }

    // 将value set到knn_result中
    knn_result.add_result(std::fabs(key - savl_tree->value), savl_tree->value);

    // 根据key去不同的子树查找
    if (key < savl_tree->value) {
        if (KNN_search_radius(savl_tree->left, key, knn_result)) {
            return true;
        }
        if (std::fabs(key - savl_tree->value) < knn_result.worst_distance()) {
            return KNN_search_radius(savl_tree->right, key, knn_result);
        } else {
            return false;
        }
    } else {
        if (KNN_search_radius(savl_tree->right, key, knn_result)) {
            return true;
        }
        if (std::fabs(key - savl_tree->value) < knn_result.worst_distance()) {
            return KNN_search_radius(savl_tree->left, key, knn_result);
        } else {
            return false;
        }
    }

    return true;
}

bool KDTreeAVLNearestNeighbors::set_data(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix, int leaf_size) {
    _kd_tree.input(input_matrix);
    _input_matrix = input_matrix;
    return _kd_tree.create_kd_tree(leaf_size);
}

bool KDTreeAVLNearestNeighbors::KNN_search_number(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultNumber& knn_result) {
    return KNN_search_number(_kd_tree.get_root(), key, knn_result);
}

bool KDTreeAVLNearestNeighbors::KNN_search_number(
    const std::shared_ptr<AAPCD::KDTreeNode>& root,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
    KNNResultNumber& knn_result) {
    if (root == nullptr) {
        return false;
    }

    if (root->is_leaf) {
        for (size_t i = 0; i < root->value_indices.size(); ++i) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> vdiff = key.col(0) - _input_matrix.col(root->value_indices[i]);
            double diff = vdiff.norm();
            knn_result.add_result(diff, root->value_indices[i]);
        }
        return true;
    }

    if (key(root->axis, 0) <= root->key) {
        KNN_search_number(root->left, key, knn_result);
        if (abs(key(root->axis, 0) - root->key) < knn_result.worst_distance()) {
            KNN_search_number(root->right, key, knn_result);
        }
    } else {
         KNN_search_number(root->right, key, knn_result);
        if (abs(key(root->axis, 0) - root->key) < knn_result.worst_distance()) {
            KNN_search_number(root->left, key, knn_result);
        }
    }
    return true;
}
