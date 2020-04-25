#include "Nearest_Neighbors/Octree.h"

int main() {
    Eigen::Vector3d a;
    a << 2, -3, -9;
    // std::cout << a.sum() << std::endl;
    std::cout << a.cwiseAbs() << std::endl;

    // Eigen::Matrix<double, 3, 4> data;
    // data << 0, 1, 0, -1,
    //         -1, 0, 1, 0,
    //         0, 0, 0, 0;
    // std::cout << "data: " << std::endl << data << std::endl;

    // AAPCD::Octree octree;
    // octree.input(data);
    // octree.build(1, 1, true);

    // ---------------------------
    // Eigen::Matrix<double, 3, 4> data;
    // data << -1, 1, -1, 1,
    //         -1, -1, 1, 1,
    //         0, 0, 0, 0;
    // std::cout << "data: " << std::endl << data << std::endl;

    // AAPCD::Octree octree;
    // octree.input(data);
    // octree.build(1, 1, true);
    return 0;
}