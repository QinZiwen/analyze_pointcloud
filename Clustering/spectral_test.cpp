#include "Clustering/spectral.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(2, 9);
    data << 1, 1, 2, 2, 3, 3, 3, 4, 4,
            1, 2, 1, 2, 1, 2, 4, 2, 3;
    std::cout << data << std::endl;

    AAPCD::Spectral spectral;
    spectral.input(data);
    spectral.compute(2);
    spectral.print_clusters();
    return 0;
}