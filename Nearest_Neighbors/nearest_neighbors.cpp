#include "Nearest_Neighbors/nearest_neighbors.h"

bool NearestNeighbors::set_data(const std::vector<double>& data) {
    _savl_tree.insert_from_vector(data);
    return true;
}

bool NearestNeighbors::KNN_search_number(double key, KNNResultNumber& knn_result) {
    return KNN_search_number(_savl_tree.get_root(), key, knn_result);
}

bool NearestNeighbors::KNN_search_number(const std::shared_ptr<SAVLNode>& savl_tree, double key, KNNResultNumber& knn_result) {
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

bool NearestNeighbors::KNN_search_radius(double key, KNNResultRadius& knn_result) {
    return KNN_search_radius(_savl_tree.get_root(), key, knn_result);
}

bool NearestNeighbors::KNN_search_radius(const std::shared_ptr<SAVLNode>& savl_tree, double key, KNNResultRadius& knn_result) {
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