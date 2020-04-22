#include "Nearest_Neighbors/KD_Tree.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(2, 4);
    data << 1, 2, 3, 4,
            1, 2, 1, 2;
    
    AAPCD::KDTree kd_tree;
    kd_tree.input(data);
    kd_tree.create_kd_tree(1);

    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data(1, 5);
    // data << 1, 2, 3, 4, 5;
    
    // AAPCD::KDTree kd_tree;
    // kd_tree.input(data);
    // kd_tree.create_kd_tree(1);
    return 0;
}