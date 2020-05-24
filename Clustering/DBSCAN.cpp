#include "Clustering/DBSCAN.h"

namespace AAPCD {

void DBSCAN::input(const Eigen::MatrixXd& input_data, int leaf_size) {
    _data = input_data;

    _kdtree_nn.set_data(input_data, leaf_size);
}

bool DBSCAN::compute(double radius, int min_points) {
    std::vector<bool> visited(_data.cols(), false);
    std::vector<bool> cluster_point(_data.cols(), false);  //防止同一个数据点的id被多次添加到cluster.neighborhood

    // 遍历所有数据
    for (size_t i = 0; i < _data.cols(); ++i) {
        if (!visited[i]) {
            std::cout << "To find i = " << i << " neighbors" << std::endl;
            // 寻找邻近点
            const Eigen::VectorXd& data_point = _data.col(i);
            std::vector<size_t> neighbors;
            if (!find_knn(data_point, radius, neighbors)) {
                std::cerr << "find_knn failure" << std::endl;
                return false;
            }

            std::cout << "i: " << i << ", neighbors: " << neighbors.size() << ", " << min_points << std::endl;
            // 判断是否为core point
            if (neighbors.size() >= min_points) {  // core point
                DBSCANCluster cluster;
                cluster.core_point.emplace_back(i);  // set core point
                cluster.neighborhood.emplace_back(i);
                cluster_point[i] = true;
                visited[i] = true;

                // add neighbors
                for (size_t k = 0; k < neighbors.size(); ++k) {
                    if (!visited[neighbors[k]] && !cluster_point[neighbors[k]]) {
                        cluster_point[neighbors[k]] = true;
                        cluster.neighborhood.emplace_back(neighbors[k]);
                    }
                }

                // 以当前点为起点，从他的neighbors中寻找其他core point
                for (size_t k = 0; k < cluster.neighborhood.size(); ++k) {
                    if (!visited[cluster.neighborhood[k]]) {
                        std::vector<size_t> nneighbors;
                        if (!find_knn(_data.col(cluster.neighborhood[k]), radius, nneighbors)) {
                            std::cerr << "find_knn failure" << std::endl;
                            return false;
                        }

                        std::cout << "k: " << k << ", nneighbors: " << nneighbors.size() << ", " << min_points << std::endl;
                        if (nneighbors.size() >= min_points) {  // 又找到一个core point
                            cluster.core_point.emplace_back(cluster.neighborhood[k]);  // set core point
                            visited[cluster.neighborhood[k]] = true;
                            cluster_point[cluster.neighborhood[k]] = true;
                            
                            // add neighbors
                            for (size_t n = 0; n < nneighbors.size(); ++n) {
                                if (!visited[nneighbors[n]] && !cluster_point[nneighbors[n]]) {
                                    cluster_point[nneighbors[n]] = true;
                                    cluster.neighborhood.emplace_back(nneighbors[n]);
                                }
                            }
                        } else {
                            // 噪声点
                            visited[cluster.neighborhood[k]] = true;
                        }
                    }
                }

                _clusters.emplace_back(cluster);
            } else {
                // 噪声点
                visited[i] = true;
            }
        }
    }
    return true;
}

bool DBSCAN::find_knn(const Eigen::VectorXd& data, double radius, std::vector<size_t>& neighbors) {
    KNNResultRadius knn_radius(radius);
    if (!_kdtree_nn.KNN_search_radius(data, knn_radius)) {
        std::cerr << "knn_search_radius failure" << std::endl;
        return false;
    }

    std::vector<DistanceValue> dis_value = knn_radius.get_distance_value();
    neighbors.clear();
    for (const auto& dv : dis_value) {
        neighbors.emplace_back(dv.value);
    }
    return true;
}

std::vector<DBSCANCluster> DBSCAN::get_clusters() {
    return _clusters;
}

bool DBSCAN::save_cluster_data_to_file(const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    for (const auto& c : _clusters) {
        for (size_t i = 0; i < c.neighborhood.size(); ++i) {
            ofs << _data(0, c.neighborhood[i]) << " " << _data(1, c.neighborhood[i]) << " ";
        }
        ofs << std::endl;
    }

    ofs.close();
    std::cout << "save result to " << file_name << std::endl;
    return true;
}

void DBSCAN::print() {
    std::cout << "===== DBSCAN clusters =====" << std::endl;
    for (const auto& c : _clusters) {
        std::cout << "core_point index: size: " << c.core_point.size() << ", ";
        for (size_t i = 0; i < c.core_point.size(); ++i) {
            std::cout << c.core_point[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "neighborhood index: size: " << c.neighborhood.size() << ", ";
        for (size_t i = 0; i < c.neighborhood.size(); ++i) {
            std::cout << c.neighborhood[i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "===========================" << std::endl;
}

}  // namespace AAPCD
