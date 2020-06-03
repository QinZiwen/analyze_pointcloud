#include "Nearest_Neighbors/Octree.h"

#include <map>
#include <cmath>

namespace AAPCD {

void Octree::input(const Eigen::Matrix<double, 3, Eigen::Dynamic>& input_matrix) {
    _input_matrix = input_matrix;
}

bool Octree::build(int leaf_size, double min_length, bool verbose) {
    _leaf_size = leaf_size;
    _min_extent = min_length / 2;
    _verbose = verbose;

    Eigen::Vector3d center_max = _input_matrix.rowwise().maxCoeff();
    Eigen::Vector3d center_min = _input_matrix.rowwise().minCoeff();
    Eigen::Vector3d center = center_min + (center_max - center_min) / 2;
    double extent = (center_max - center_min).maxCoeff() / 2;

    std::vector<u_int> value_indices(_input_matrix.cols());
    for (int i = 0; i < _input_matrix.cols(); ++i) {
        value_indices[i] = i;
    }
    return octree_recursive_build(_root, _input_matrix, center, value_indices, extent);
}

bool Octree::octree_recursive_build(
    std::shared_ptr<Octant>& root,
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& values,
    const Eigen::Vector3d& center,
    const std::vector<u_int>& value_indices,
    double extent) {
    if (value_indices.size() == 0) {
        return false;
    }

    if (_verbose) {
        std::cout << "center: " << center.transpose() << std::endl;
        std::cout << "value_indices size: " << value_indices.size() << std::endl;
        std::cout << "extent: " << extent << std::endl;
    }
    if (root == nullptr) {
        root.reset(new Octant(center, extent, value_indices, false));
    }

    if (value_indices.size() <= _leaf_size || extent <= _min_extent) {
        root->is_leaf = true;
        
        if (_verbose) {
            std::cout << "leaf value: " << std::endl;
            for (int i = 0; i < value_indices.size(); ++i) {
                std::cout << values.col(value_indices[i]).transpose() << std::endl;
            }
            std::cout << std::endl;
        }
    } else {
        // recursive build
        std::map<u_char, std::vector<u_int>> children_value_indices;

        // determine which child a value belongs to
        for (const u_int& vi : value_indices) {
            const Eigen::Vector3d& value_tmp = values.col(vi);
            u_char morton_code = 0;
            if (value_tmp(0) > center(0)) {
                morton_code |= 1;
            }
            if (value_tmp(1) > center(1)) {
                morton_code |= 2;
            }
            if (value_tmp(2) > center(2)) {
                morton_code |= 4;
            }
            children_value_indices[morton_code].emplace_back(vi);
        }

        // create child
        double factor[2] = {-0.5, 0.5};
        for (int i = 0; i < 8; ++i) {
            // determin child certer and extent
            Eigen::Vector3d child_center;
            child_center(0) = center(0) + factor[int((i & 1) > 0)] * extent;
            child_center(1) = center(1) + factor[int((i & 2) > 0)] * extent;
            child_center(2) = center(2) + factor[int((i & 4) > 0)] * extent;
            double child_extent = 0.5 * extent;
            octree_recursive_build(
                root->children[i], values, child_center, children_value_indices[i], child_extent);
        }
    }
    return true;
}

bool Octree::inside(
    const Eigen::Vector3d& query,
    const std::shared_ptr<Octant>& octant,
    double radius) {
    Eigen::Vector3d query_offset = (query - octant->center).cwiseAbs();
    Eigen::Vector3d possible_space = query_offset + Eigen::Vector3d(radius, radius, radius);
    if (possible_space(0) < octant->extent &&
        possible_space(1) < octant->extent &&
        possible_space(2) < octant->extent) {
        return true;
    } else {
        return false;
    }
}

bool Octree::overlap(
    const Eigen::Vector3d& query,
    const std::shared_ptr<Octant>& octant,
    double radius) {
    Eigen::Vector3d query_offset = (query - octant->center).cwiseAbs();
    double max_dist = radius + octant->extent;
    // case 1
    for (int i = 0; i < 3; ++i) {
        if (query_offset(i) > max_dist) {
            return false;
        }
    }

    // case 2: contacting the face of the octant
    Eigen::Vector3d is_contact_face;
    for (int i = 0; i < 3; ++i) {
        is_contact_face(i) = query_offset(i) < octant->extent ? 1 : 0;
    }
    if (is_contact_face.sum() >= 2) {  // =2 表示query ball与立方体相贴，=3 表示相交
        return true;
    }

    // case 3: query ball is contacting the edge or corner of octant
    double x_diff = std::max(query_offset(0) - octant->extent, 0.);
    double y_diff = std::max(query_offset(1) - octant->extent, 0.);
    double z_diff = std::max(query_offset(2) - octant->extent, 0.);

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius;
}

std::shared_ptr<Octant> Octree::get_root() {
    return _root;
}

}  // namespace AAPCD