#include "Nearest_Neighbors/KNN_result.h"

int main() {
    std::vector<DistanceValue> data = {
        DistanceValue(3, 3),
        DistanceValue(4, 4),
        DistanceValue(1, 1),
        DistanceValue(5, 5),
        DistanceValue(6, 6),
        DistanceValue(0, 0),
        DistanceValue(-1, -1),
        DistanceValue(2, 2),
        DistanceValue(9, 9)
    };

    KNNResultNumber knn_result(4);
    for (const DistanceValue dv : data) {
        knn_result.add_result(dv.distance, dv.value);
    }
    std::cout << "size: " << knn_result.size() << std::endl;
    std::cout << "is full: " << knn_result.is_full() << std::endl;
    std::cout << "worst distance: " << knn_result.worst_distance() << std::endl;
    knn_result.print();
    return 0;
}