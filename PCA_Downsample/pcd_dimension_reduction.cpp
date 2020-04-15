#include <iostream>
#include <string>
#include <vector>

#include "PCA_Downsample/pca.h"
#include "utils/utils.hpp"

int main(int argc, char** argv) {
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& pcd = read_pointcloud_from_file(argv[1]);
    std::cout << "point cloud size: " << pcd.cols() << std::endl;

    PCA pca;
    pca.input(pcd);
    pca.compute(PCA::eigen_vector_order::DESCENDING);
    std::cout << "eigen values: \n" << pca.get_eigen_values() << std::endl;
    std::cout << "eigen vector: \n" << pca.get_eigen_vector() << std::endl;
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_vector = pca.get_eigen_vector();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_1 = pcd.transpose() * eigen_vector.col(0);
    if (!save_pointcloud_to_file(eigen_1, "eigen_1.txt")) {
        std::cerr << "save_pointcloud_to_file failed" << std::endl;
        return 1;
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_2 = pcd.transpose() * eigen_vector.leftCols(2);
    if (!save_pointcloud_to_file(eigen_2, "eigen_2.txt")) {
        std::cerr << "save_pointcloud_to_file failed" << std::endl;
        return 1;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_3 = pcd.transpose() * eigen_vector;
    if (!save_pointcloud_to_file(eigen_3, "eigen_3.txt")) {
        std::cerr << "save_pointcloud_to_file failed" << std::endl;
        return 1;
    }
    return 0;
}