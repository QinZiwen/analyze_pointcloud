#include <iostream>
#include <string>
#include <vector>

#include "PCA_Downsample/pca.h"

int main(int argc, char** argv) {
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& pcd = Utils::read_pointcloud_from_file(argv[1]);
    std::cout << "point cloud size: " << pcd.cols() << std::endl;

    PCA pca;
    pca.input(pcd);
    pca.compute(PCA::eigen_vector_order::DESCENDING);
    std::cout << "eigen values: \n" << pca.get_eigen_values() << std::endl;
    std::cout << "eigen vector: \n" << pca.get_eigen_vector() << std::endl;
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& eigen_vector = pca.get_eigen_vector();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_1 = pcd.transpose() * eigen_vector.col(0);
    // Reconstruct, x^T * z = a  --> x^T = a * z^T
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> recons_1 = eigen_1 * eigen_vector.col(0).transpose();
    if (!Utils::save_pointcloud_to_file(recons_1, "eigen_1.txt")) {
        std::cerr << "save_pointcloud_to_file failed" << std::endl;
        return 1;
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_2 = pcd.transpose() * eigen_vector.leftCols(2);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> recons_2 = eigen_2 * eigen_vector.leftCols(2).transpose();
    if (!Utils::save_pointcloud_to_file(recons_2, "eigen_2.txt")) {
        std::cerr << "save_pointcloud_to_file failed" << std::endl;
        return 1;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_3 = pcd.transpose() * eigen_vector;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> recons_3 = eigen_3 * eigen_vector.transpose();
    if (!Utils::save_pointcloud_to_file(eigen_3, "eigen_3.txt")) {
        std::cerr << "save_pointcloud_to_file failed" << std::endl;
        return 1;
    }
    return 0;
}