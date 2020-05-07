#include "Clustering/KMeans.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(2, 5);
    data << 1, 1, 3, 4, 4,
            1, 2, 4, 2, 3;
    std::cout << data << std::endl;

    AAPCD::KMeans kmeans;
    kmeans.input(data);
    kmeans.compute(2);
    std::vector<AAPCD::Cluster> clusters = kmeans.get_clusters();

    for (size_t c = 0; c < clusters.size(); ++c) {
        std::cout << "cluster: " << clusters[c].id << ", " << clusters[c].center.transpose() << std::endl;
        for (size_t i = 0; i < clusters[c].data_index_size; ++i) {
            std::cout << data.col(clusters[c].data_index[i]).transpose() << std::endl;
        }
    }
    return 0;
}