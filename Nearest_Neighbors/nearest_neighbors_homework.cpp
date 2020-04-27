#include "utils/utils.hpp"

#include "Nearest_Neighbors/nearest_neighbors.h"
#include "KD_tree/KD_tree.hpp"
#include <sys/time.h>

int main(int argc, char** argv) {
    std::string bin_file(argv[1]);

    struct timeval process_start;
    struct timeval process_end;
    double process_timer;

    gettimeofday(&process_start, NULL);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pcd;
    Utils::read_point_cloud_from_bin(bin_file, pcd);
    std::cout << "pcd size: " << pcd.cols() << std::endl;
    gettimeofday(&process_end, NULL);
    process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
    std::cout << "read pcd time: " << process_timer << " s" << std::endl;

    // param
    int k_index = 1000;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> key(3, 1);
    key = pcd.col(k_index);
    double radius = 10;

    // brute-force
    gettimeofday(&process_start, NULL);
    KNNResultRadius BF_knn_result(radius);
    for (size_t i = 0; i < pcd.cols(); ++i) {
        Eigen::Vector3d dis = key.col(0) - pcd.col(i);
        BF_knn_result.add_result(dis.norm(), i);
    }
    // BF_knn_result.print();
    std::cout << "brute-force find size: " << BF_knn_result.size() << std::endl;
    gettimeofday(&process_end, NULL);
    process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
    std::cout << "brute-force time: " << process_timer << " s" << std::endl;

    // third kd-tree
    // transformat data
    pointVec points;
    for (size_t i = 0; i < pcd.cols(); ++i) {
        const auto& tmp = pcd.col(i);
        point_t pt = {tmp(0, 0), tmp(1, 0), tmp(2, 0)};
        // std::cout << pt[0] << " " << pt[1] << " " << pt[2] << std::endl;
        points.emplace_back(pt);
    }
    std::cout << "points size: " << points.size() << std::endl;
    point_t pt_key = {key(0, 0), key(1, 0), key(2, 0)};

    KDTree tree(points);
    gettimeofday(&process_start, NULL);
    auto res2 = tree.neighborhood_points(pt_key, radius);
    std::cout << "third kd-tree find size: " << res2.size() << std::endl;
    gettimeofday(&process_end, NULL);
    process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
    std::cout << "third kd-tree time: " << process_timer << " s" << std::endl;

    // my kd-tree
    KDTreeAVLNearestNeighbors kd_tree_nn;
    KNNResultRadius kd_tree_knn_result_rad(radius);
    kd_tree_nn.set_data(pcd, 200);

    gettimeofday(&process_start, NULL);
    kd_tree_nn.KNN_search_radius(key, kd_tree_knn_result_rad);
    // kd_tree_knn_result_rad.print();
    std::cout << "my kd-tree find size: " << kd_tree_knn_result_rad.size() << std::endl;
    gettimeofday(&process_end, NULL);
    process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
    std::cout << "my kd-tree time: " << process_timer << " s" << std::endl;

    // my octree
    OctreeAVLNearestNeighbors octree_nn;
    octree_nn.set_data(pcd, 200, 10);

    KNNResultRadius octree_knn_result_rad(radius);
    gettimeofday(&process_start, NULL);
    octree_nn.KNN_search_radius(key, octree_knn_result_rad);
    // octree_knn_result_rad.print();
    std::cout << "my octree find size: " << octree_knn_result_rad.size() << std::endl;
    gettimeofday(&process_end, NULL);
    process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
    std::cout << "my octree time: " << process_timer << " s" << std::endl;

    return 0;
} 