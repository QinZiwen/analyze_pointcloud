#include "Nearest_Neighbors/nearest_neighbors.h"

int main() {
    std::vector<double> data = {1, 3, 2, 4, 5, 6, 7, 8, 9};

    double k = 4;
    KNNResultNumber knn_result(5);

    NearestNeighbors nn;
    nn.set_data(data);
    nn.KNN_search_number(k, knn_result);
    knn_result.print();

    KNNResultRadius knn_result_rad(1);
    nn.KNN_search_radius(k, knn_result_rad);
    knn_result_rad.print();

    return 0;
}