#pragma once

#include "utils/utils.hpp"
#include "Nearest_Neighbors/simple_AVLTree.h"
#include "Nearest_Neighbors/KD_Tree.h"
#include "Nearest_Neighbors/Octree.h"
#include "Nearest_Neighbors/KNN_result.h"

class AVLNearestNeighbors {
public:
    bool set_data(const std::vector<double>& data);
    bool KNN_search_number(double key, KNNResultNumber& knn_result);
    bool KNN_search_radius(double key, KNNResultRadius& knn_result);

private:
    bool KNN_search_number(const std::shared_ptr<SAVLNode>& savl_tree, double key, KNNResultNumber& knn_result);
    bool KNN_search_radius(const std::shared_ptr<SAVLNode>& savl_tree, double key, KNNResultRadius& knn_result);

private:
    SAVLTree _savl_tree;
};

class KDTreeAVLNearestNeighbors {
public:
    bool set_data(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix, int leaf_size);
    bool KNN_search_number(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultNumber& knn_result);

    bool KNN_search_radius(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultRadius& knn_result);

private:
    bool KNN_search_number(
        const std::shared_ptr<AAPCD::KDTreeNode>& root,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultNumber& knn_result);

    bool KNN_search_radius(
        const std::shared_ptr<AAPCD::KDTreeNode>& root,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultRadius& knn_result);

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _input_matrix;
    AAPCD::KDTree _kd_tree;
};

class OctreeAVLNearestNeighbors {
public:
    bool set_data(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& input_matrix,
        int leaf_size,
        double min_length);
    bool KNN_search_number(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultNumber& knn_result);

    bool KNN_search_radius(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultRadius& knn_result);

private:
    bool KNN_search_number(
        const std::shared_ptr<AAPCD::Octant>& root,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultNumber& knn_result);

    bool KNN_search_radius(
        const std::shared_ptr<AAPCD::Octant>& root,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& key,
        KNNResultRadius& knn_result);

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _input_matrix;
    AAPCD::Octree _octree;
};
