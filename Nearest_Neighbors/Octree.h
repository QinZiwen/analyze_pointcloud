#pragma

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
} Octant;

class Octree {
public:
    void input(const Eigen::Matrix<double, 3, Eigen::Dynamic>& input_matrix);
    bool build(double leaf_size, double min_length, bool verbose = false);

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
    double _leaf_size;
    double _min_extent;

    bool _verbose;
};

}  // namespace AAPCD