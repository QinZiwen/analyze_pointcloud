#include "Fitting/find_ground.h"

#include <stack>
#include <ctime>
#include <set>

namespace AAPCD {
void FindGround::input(const Eigen::MatrixXd& input_matrix) {
    _input_data = input_matrix;
}

bool FindGround::find(const METHOD& method) {
    switch (method) {
    case METHOD::FG_OCTREE:
        return find_use_octree();
    default:
        std::cerr << "Invalid method" << std::endl;
        return false;
    }
    return true;
}

bool FindGround::find_use_octree() {
    const bool is_downsample = false;
    const double vgd_grid_size = 1;
    const double octree_leaf_size = 30;
    const double octree_min_length = 10;

    // downsample
    if (is_downsample) {
        std::cout << "downsampe before size: " << _input_data.cols() << std::endl;
        VoxelGridDownsampling vgd;
        vgd.set_data(_input_data);
        vgd.downsampling(vgd_grid_size);
        const std::vector<std::pair<int, Eigen::Vector3d>>& vgd_res = vgd.get_downsampel_result();
        _data.resize(3, vgd_res.size());
        for (size_t i = 0; i < vgd_res.size(); ++i) {
            _data.col(i) = vgd_res[i].second;
        }
        std::cout << "downsample after size: " << _data.cols() << std::endl;
    } else {
        _data = _input_data;
    }

    // 1. create octree
    Octree octree;
    octree.input(_data);
    if (!octree.build(octree_leaf_size, octree_min_length)) {
        std::cout << "build octree failure" << std::endl;
        return false;
    }

    std::shared_ptr<Octant> octants = octree.get_root();

    if (!find_ground_direction(octants)) {
        std::cerr << "find_ground_direction failed" << std::endl;
        return false;
    }

    // 4. collect inlier using plane direction
    // _save_octree_ofs.open("octree_data_with_color.txt");
    // if (!_save_octree_ofs.is_open()) {
    //     std::cerr << "Can not open octree_data_with_color.txt" << std::endl;
    //     return false;
    // }
    // srand(time(0));

    // for (size_t i = 0; i < _data.cols(); ++i) {
    //     if (distance_plane(_data.col(i)) < 0.5) {
    //         _save_octree_ofs << _data(0, i) << " "
    //                     << _data(1, i) << " "
    //                     << _data(2, i) << " "
    //                     << std::to_string(1) << " "
    //                     << std::to_string(0) << " "
    //                     << std::to_string(0) << std::endl;
    //     }
    // }

    // _save_octree_ofs.close();

    // 5. cluster ground from inlier using DBSCAN
    return true;
}

bool FindGround::find_ground_direction(const std::shared_ptr<Octant>& octants) {
    std::stack<std::shared_ptr<struct Octant>> stack;
    for (int i = 0; i < 8; ++i) {
        if (octants->children[i] != nullptr) {
            stack.emplace(octants->children[i]);
        }
    }

    std::vector<Eigen::VectorXd> plane_param_vec;
    // 2. fitting the plane in each cube
    while (!stack.empty()) {
        std::shared_ptr<struct Octant> oct = stack.top();
        // oct->print();
        stack.pop();

        // insert new child
        for (int i = 0; i < 8; ++i) {
            if (oct->children[i] != nullptr) {
                stack.emplace(oct->children[i]);
            }
        }

        // test save every octant as random color
        if (_save_octree_to_file && oct->is_leaf == true && _save_octree_with_color == true) {
            // oct->print();
            float r = float(rand() % 100) / 100;
            float g = float(rand() % 100) / 100;
            float b = float(rand() % 100) / 100;

            for (size_t i = 0; i < oct->value_indices.size(); ++i) {
                _save_octree_ofs << _data(0, oct->value_indices[i]) << " "
                        << _data(1, oct->value_indices[i]) << " "
                        << _data(2, oct->value_indices[i]) << " "
                        << std::to_string(r) << " "
                        << std::to_string(g) << " "
                        << std::to_string(b) << std::endl;
            }
        }

        // fitting the plane
        if (oct->value_indices.size() > 2 && oct->is_leaf == true) {
            Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, oct->value_indices.size());
            for (size_t i = 0; i < oct->value_indices.size(); ++i) {
                points.col(i) = _data.col(oct->value_indices[i]);
            }
            // std::cout << "points: " << points << std::endl;

            if (!is_share_plane(points)) {
                continue;
            }
            std::cout << "fitting plane points size: " << points.cols() << std::endl;
            
            Eigen::VectorXd plane_param;  // Ax + By + Cz + D = 0
            // double N = log(1 - 0.99) / log(1 - pow(0.99, 6));
            // std::cout << "num RANSAC iter: " << N << std::endl;
            int min_num = int(points.cols() * 0.1) > 3 ? int(points.cols() * 0.1) : 3;
            std::cout << "min_num: " << min_num << std::endl;
            if (!fitting_plane_RANSAC(points, min_num, 35, plane_param)) {
                std::cerr << "fitting_plane_RANSAC failed" << std::endl;
                return false;
            } else {
                // save plane normal
                if (_save_octree_to_file && _save_ground_normal) {
                    for (size_t i = 0; i < oct->value_indices.size(); ++i) {
                        _save_octree_ofs << _data(0, oct->value_indices[i]) << " "
                                << _data(1, oct->value_indices[i]) << " "
                                << _data(2, oct->value_indices[i]) << " "
                                << std::to_string(plane_param[0]) << " "
                                << std::to_string(plane_param[1]) << " "
                                << std::to_string(plane_param[2]) << std::endl;
                    }
                }
            }
            plane_param_vec.emplace_back(plane_param);
        }
    }

    std::cout << "plane_param_vec size: " << plane_param_vec.size() << std::endl;
    // 3. find plane direction
    if (!cluster_plane_param(plane_param_vec, _ground_param)) {
        std::cerr << "cluster_plane_param failed" << std::endl;
        return false;
    }
    std::cout << "_ground_param: " << _ground_param.transpose() << std::endl;


    if (_save_octree_to_file && _save_ground_normal) {
        // save ground direction
        _save_octree_ofs << "0 0 0 "
                            << std::to_string(_ground_param[0]) << " "
                            << std::to_string(_ground_param[1]) << " "
                            << std::to_string(_ground_param[2]) << std::endl;

        _save_octree_ofs << "-1 -1 1 "
                            << std::to_string(_ground_param[0]) << " "
                            << std::to_string(_ground_param[1]) << " "
                            << std::to_string(_ground_param[2]) << std::endl;
        _save_octree_ofs << "-1 1 1 "
                            << std::to_string(_ground_param[0]) << " "
                            << std::to_string(_ground_param[1]) << " "
                            << std::to_string(_ground_param[2]) << std::endl;
        _save_octree_ofs << "1 -1 1 "
                            << std::to_string(_ground_param[0]) << " "
                            << std::to_string(_ground_param[1]) << " "
                            << std::to_string(_ground_param[2]) << std::endl;
        _save_octree_ofs << "1 1 1 "
                            << std::to_string(_ground_param[0]) << " "
                            << std::to_string(_ground_param[1]) << " "
                            << std::to_string(_ground_param[2]) << std::endl;
    }

    return true;
}

bool FindGround::fitting_plane_RANSAC(const Eigen::MatrixXd& points, int min_num, int max_iter, Eigen::VectorXd& best_param) {
    srand(time(0));
    if (min_num < 3) {
        std::cerr << "It is estimated that the plane needs at least 3 points" << std::endl;
        return false;
    }

    size_t num_point = points.cols();
    best_param = Eigen::VectorXd::Zero(4);
    double best_inlier_ratio = 0;
    for (int i = 0; i < max_iter; ++i) {
        // min subset
        std::set<int> subset_idx;
        while (subset_idx.size() < min_num) {
            int idx = rand() % num_point;
            subset_idx.emplace(idx);
            // std::cout << "idx: " << idx << std::endl;
        }

        Eigen::MatrixXd subset = Eigen::MatrixXd::Zero(3, min_num);
        int subset_id = 0;
        for (const auto s : subset_idx) {
            // for (int row = 0; row < 3; ++row) {
            //     subset(row, subset_id) = points(row, s);
            // }
            subset.col(subset_id) = points.col(s);
            ++subset_id;
        }
        // std::cout << "subset: \n" << subset << std::endl;

        // estimate normal of plane
        PCA pca;
        pca.input(subset);
        if (!pca.compute(PCA::eigen_vector_order::ASCENDING)) {
            std::cerr << "PCA compute failed" << std::endl;
            return false;
        }
        Eigen::MatrixXd eigen_values = pca.get_eigen_values();
        // std::cout << "eigen_values: \n" << eigen_values << std::endl;

        const Eigen::MatrixXd&  eigen_vector = pca.get_eigen_vector();
        Eigen::VectorXd est_normal = eigen_vector.col(0);

        double D = -1 * (subset.adjoint() * est_normal).mean();

        double inlier_ratio = 0;
        for (int t = 0; t < num_point; ++t) {
            double dis = abs(est_normal.transpose() * points.col(t) + D) / est_normal.norm();
            if (dis < 0.001) {
                ++inlier_ratio;
            }
        }
        inlier_ratio /= num_point;
        std::cout << "inlier_ratio: " << inlier_ratio << std::endl;

        // inlier 满足阈值
        if (inlier_ratio > 0.95) {
            best_inlier_ratio = inlier_ratio;
            for (int b = 0; b < 3; ++b) {
                best_param(b) = est_normal(b);
            }
            best_param(3) = D;
            break;
        }

        // 如果没有满足阈值，则保留最好的。
        if (inlier_ratio > best_inlier_ratio) {
            best_inlier_ratio = inlier_ratio;
            for (int b = 0; b < 3; ++b) {
                best_param(b) = est_normal(b);
            }
            best_param(3) = D;
        }
    }
    return true;
}

bool FindGround::is_share_plane(const Eigen::MatrixXd& points) {
    if (points.cols() < 3) {
        return false;
    }

    for (int i = 0; i < points.cols() - 2; ++i) {
        for (int j = i + 1; j < points.cols() - 1; ++j) {
            for (int k = j + 1; k < points.cols(); ++k) {
                Eigen::VectorXd d1 = points.col(i) - points.col(j);
                Eigen::VectorXd d2 = points.col(i) - points.col(k);
                
                double diff = double(d1.transpose() * d2) / (d1.norm() * d2.norm());
                if (diff - 0 < 0.1) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool FindGround::cluster_plane_param(const std::vector<Eigen::VectorXd>& plane_param_vec, Eigen::VectorXd& param) {
    if (plane_param_vec.size() <= 0) {
        return false;
    }

    // Eigen::MatrixXd points = Eigen::MatrixXd::Zero(4, plane_param_vec.size());
    Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, plane_param_vec.size());
    for (size_t i = 0; i < plane_param_vec.size(); ++i) {
        // points.col(i) = plane_param_vec[i];
        for (size_t row = 0; row < 3; ++row) {
            points(row, i) = plane_param_vec[i][row];
        }
    }

    KMeans kmeans;
    kmeans.input(points);
    kmeans.compute(3);
    std::vector<Cluster> cluster = kmeans.get_clusters();

    double num_cls = 0;
    for (size_t i = 0; i < cluster.size(); ++i) {
        if (cluster[i].data_index_size > num_cls) {
            param = cluster[i].center;
        }
    }
    return true;
}

double FindGround::distance_plane(const Eigen::VectorXd& point) {
    double dis = 0;
    Eigen::VectorXd normal = _ground_param.topRows(3);

    // dis = abs(double(normal.transpose() * point) + _ground_param(3)) / normal.norm();

    Eigen::VectorXd p = _data.col(0) - point;
    Eigen::VectorXd p_norm = p / p.sum();

    Eigen::VectorXd tmp_normal = normal / normal.sum();
    dis = abs(double(tmp_normal.transpose() * p_norm) - 1);

    std::cout << "dis: " << dis << std::endl;
    return dis;
}

}  // namespace AAPCD