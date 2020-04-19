#pragma once

#include "utils/utils.hpp"
#include "Nearest_Neighbors/simple_AVLTree.h"
#include "Nearest_Neighbors/KNN_result.h"

class NearestNeighbors {
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
