#include "Clustering/GMM.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(2, 5);
    data << 1, 1, 3, 5, 5,
            1, 2, 4, 2, 3;
    std::cout << data << std::endl;

    AAPCD::GMM gmm;
    gmm.input(data);
    gmm.compute(3, 5);
    gmm.print_clusters();
    gmm.print_Z_nk();
}