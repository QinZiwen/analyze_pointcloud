#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "utils/utils.hpp"

namespace AAPCD {

typedef struct Octant {
    Eigen::Vector3d center;
    double extent;
    std::vector<u_int> value_indices;
    bool is_leaf;
    std::shared_ptr<struct Octant> children[8];

    Octant() {}
    Octant(
        const Eigen::Vector3d& center_,
        double extent_,
        const std::vector<u_int>& value_indices_,
        bool is_leaf_)
    : center(center_),
      extent(extent_),
      value_indices(value_indices_),
      is_leaf(is_leaf_) {}
    
    void print() {
        std::cout << "center: " << center.transpose() << ", extent: " << extent
                << ", value_indices: " << value_indices.size() << ", is_leaf: " << is_leaf << std::endl;
    }
} Octant;

class Octree {
public:
    void input(const Eigen::Matrix<double, 3, Eigen::Dynamic>& input_matrix);
    bool build(int leaf_size, double min_length, bool verbose = false);

    static bool inside(
        const Eigen::Vector3d& query,
        const std::shared_ptr<Octant>& octant,
        double radius);
    
    static bool overlap(
        const Eigen::Vector3d& query,
        const std::shared_ptr<Octant>& octant,
        double radius);

    std::shared_ptr<Octant> get_root();
private:
    bool octree_recursive_build(
        std::shared_ptr<Octant>& root,
        const Eigen::Matrix<double, 3, Eigen::Dynamic>& values,
        const Eigen::Vector3d& center,
        const std::vector<u_int>& value_indices,
        double extent);

private:
    Eigen::Matrix<double, 3, Eigen::Dynamic> _input_matrix;   // column is feature
    std::shared_ptr<Octant> _root;
    int _leaf_size;
    double _min_extent;

    bool _verbose;
};

}  // namespace AAPCD