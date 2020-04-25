#include "Nearest_Neighbors/nearest_neighbors.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(3, 4);
    data << 1, 2, 3, 4,
            1, 2, 1, 2,
            0, 0, 0, 0;
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> k(3, 1);
    k << 2.5, 1.5, 0;
    KNNResultNumber knn_result(5);

    OctreeAVLNearestNeighbors nn;
    nn.set_data(data, 1, 1);
    nn.KNN_search_number(k, knn_result);
    knn_result.print();

    KNNResultRadius knn_result_rad(1);
    nn.KNN_search_radius(k, knn_result_rad);
    knn_result_rad.print();

    // -----------------------
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(2, 4);
    // data << 1, 2, 3, 4,
    //         1, 2, 1, 2;
    
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> k(2, 1);
    // k << 2.5, 1.5;
    // KNNResultNumber knn_result(5);

    // KDTreeAVLNearestNeighbors nn;
    // nn.set_data(data, 1);
    // nn.KNN_search_number(k, knn_result);
    // knn_result.print();

    // KNNResultRadius knn_result_rad(1);
    // nn.KNN_search_radius(k, knn_result_rad);
    // knn_result_rad.print();

    // -----------------------
    // std::vector<double> data = {1, 3, 2, 4, 5, 6, 7, 8, 9};

    // double k = 4;
    // KNNResultNumber knn_result(5);

    // AVLNearestNeighbors nn;
    // nn.set_data(data);
    // nn.KNN_search_number(k, knn_result);
    // knn_result.print();

    // KNNResultRadius knn_result_rad(1);
    // nn.KNN_search_radius(k, knn_result_rad);
    // knn_result_rad.print();

    return 0;
}