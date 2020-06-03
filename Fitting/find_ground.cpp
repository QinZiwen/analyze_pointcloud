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
    if (!collect_inlier_using_plane_direction()) {
        std::cerr << "collect_inlier_using_plane_direction failed" << std::endl;
        return false;
    }

    // 5. cluster ground from inlier using DBSCAN
    // 原来想的是用DBSCAN来refine结果，但是最后觉得不太适合激光点云数据，因为激光点云进近处密，远处疏。
    // 用RANSAC refine结果，具体：
    // 1. 估计平面方程的参数
    // 2. 根据距离筛选出距离平面近的点作为ground点
    if (!refine_ground_point()) {
        std::cout << "refine_ground_point faild to run" << std::endl;
        return false;
    }

    std::cout << "find_use_octree done" << std::endl;
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
        // std::cout << "inlier_ratio: " << inlier_ratio << std::endl;

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

    // std::cout << "dis: " << dis << std::endl;
    return dis;
}

bool FindGround::collect_inlier_using_plane_direction() {
    // 每一个点利用临域内的点估计法向量，然后与估计的地面法向量计算夹角，夹角比较小的归位inlire
    OctreeAVLNearestNeighbors nn;
    nn.set_data(_data, octree_leaf_size, octree_min_length);

    _inlier_idx.clear();
    for (size_t i = 0; i < _data.cols(); ++i) {
        Eigen::VectorXd point = _data.col(i);

        KNNResultRadius knn_res(_knn_radius);
        nn.KNN_search_radius(point, knn_res);

        std::vector<DistanceValue> dv = knn_res.get_distance_value();
        Eigen::VectorXd est_normal = Eigen::VectorXd::Zero(3);
        if (dv.size() > 2) {
            // estimate normal
            Eigen::MatrixXd neig = Eigen::MatrixXd::Zero(3, dv.size());
            for (size_t k = 0; k < dv.size(); ++k) {
                neig.col(k) = _data.col(dv[k].value);
            }

            PCA pca;
            pca.input(neig);
            if (!pca.compute(PCA::eigen_vector_order::ASCENDING)) {
                std::cerr << "PCA compute failed" << std::endl;
                return false;
            }
            Eigen::MatrixXd eigen_values = pca.get_eigen_values();
            // std::cout << "eigen_values: \n" << eigen_values << std::endl;

            const Eigen::MatrixXd&  eigen_vector = pca.get_eigen_vector();
            est_normal = eigen_vector.col(0);
            // std::cout << "est_normal: " << est_normal.transpose() << std::endl;

            Eigen::VectorXd ground_normal = _ground_param.topRows(3);
            // std::cout << "ground_normal: " << ground_normal.transpose() << std::endl;
            double dis = abs(double(ground_normal.transpose() * est_normal) / (ground_normal.norm() * est_normal.norm()));
            // std::cout << "dis: " << dis << std::endl;

            if (abs(dis - 1) < 0.05) {
                _inlier_idx.emplace_back(i);
            }
        }

        std::cout << "collect_inlier_using_plane_direction schedule: "
                << i << ":" << _data.cols() << " = " << double(i) / _data.cols() << std::endl;
    }

    std::cout << "collect_inlier_using_plane_direction inlier size: " << _inlier_idx.size()
            << ", ratio: " << double(_inlier_idx.size()) / _data.cols() << std::endl;

    // save inlier
    if (_save_octree_to_file && _save_collect_inlier_using_plane_direction) {
        for (size_t i = 0; i < _data.cols(); ++i) {
            bool is_inlier = false;
            for (const auto& in : _inlier_idx) {
                if (i == in) {
                    is_inlier = true;
                    break;
                }
            }

            _save_octree_ofs << _data(0, i) << " "
                            << _data(1, i) << " "
                            << _data(2, i) << " ";
            if (is_inlier) {   // red
                _save_octree_ofs << std::to_string(1) << " "
                            << std::to_string(0) << " "
                            << std::to_string(0) << std::endl;
            } else {  // gree
                _save_octree_ofs << std::to_string(0) << " "
                            << std::to_string(1) << " "
                            << std::to_string(0) << std::endl;
            }
        }
    }
    return true;
}

bool FindGround::refine_ground_point() {
    if (_inlier_idx.size() < 10) {
        std::cerr << "Inlier too few" << std::endl;
        return false;
    }

    Eigen::MatrixXd inlier_data = Eigen::MatrixXd::Zero(3, _inlier_idx.size());
    for (size_t i = 0; i < _inlier_idx.size(); ++i) {
        inlier_data.col(i) = _data.col(_inlier_idx[i]);
    }
    std::cout << "inlier to refine: " << inlier_data.cols() << std::endl;

    if (!fitting_plane_RANSAC(inlier_data, 10, 35, _ground_param)) {
        std::cerr << "fitting_plane_RANSAC failed" << std::endl;
        return false;
    }
    std::cout << "ground param after refine: " << _ground_param.transpose() << std::endl;

    if (_save_octree_to_file && _save_refine_ground_point) {
        std::cout << "save result to file, ground is red, otherwise is green ..." << std::endl;

        for (size_t i = 0; i < _data.cols(); ++i) {
            bool is_inlier = false;
            Eigen::VectorXd point = _data.col(i);
            Eigen::VectorXd ground_norm = _ground_param.topRows(3);

            double dis = double(abs(ground_norm.transpose() * point + _ground_param[3]))
                    / (ground_norm.norm() * point.norm());
            if (dis < 0.05) {
                is_inlier = true;
            }

            // save to file
            _save_octree_ofs << _data(0, i) << " "
                                << _data(1, i) << " "
                                << _data(2, i) << " ";
            if (is_inlier) {   // red
                _save_octree_ofs << std::to_string(1) << " "
                            << std::to_string(0) << " "
                            << std::to_string(0) << std::endl;
            } else {  // gree
                _save_octree_ofs << std::to_string(0) << " "
                            << std::to_string(1) << " "
                            << std::to_string(0) << std::endl;
            }
        }
    }
    return true;
}

}  // namespace AAPCD