#include "PCA_Downsample/normal_estimator.h"

#include <fstream>

#include "PCA_Downsample/pca.h"

bool NormalEstimator::set_data(const std::string& pts_file) {
    std::ifstream ifs(pts_file);
    if (!ifs.is_open()) {
        std::cerr << "[FATAL ERROR] can not open " << pts_file << std::endl;
        return false;
    }

    int pcd_size = 0;
    std::string line;
    point_t pt;
    while(std::getline(ifs, line)) {
        if (line.length() == 0 || line[0] == '#') {
            continue;
        }

        std::vector<std::string> line_splits = Utils::regexsplit(line, ",");
        if (line_splits.size() < 3) {
            std::cerr << "[FATAL ERROR] input file format error" << std::endl;
            exit(1);
        }

        pt = {std::stod(line_splits[0]), std::stod(line_splits[1]), std::stod(line_splits[2])};
        _points.push_back(pt);

        ++pcd_size;
    }
    ifs.close();
    std::cout << "pointcloud size: " << pcd_size << std::endl;

    _kd_tree.set_points(_points);

    return true;
}

bool NormalEstimator::compute(double threshold, const std::string& output_file_name) {
    std::ofstream ofs;
    if (output_file_name.length() > 0) {
        ofs.open(output_file_name, std::ios::out);
        if (!ofs.is_open()) {
            std::cerr << "Can not open " << output_file_name << std::endl;
            return false;
        }
    }
    for (size_t i = 0; i < _points.size(); ++i) {
        auto neigh = _kd_tree.neighborhood_points(_points[i], threshold);
        std::cout << i << "/" << _points.size() << " neigh size: " << neigh.size() << std::endl;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> neigh_mat(3, neigh.size());
        for (int j = 0; j < neigh.size(); ++j) {
            neigh_mat(0, j) = neigh[j][0];
            neigh_mat(1, j) = neigh[j][1];
            neigh_mat(2, j) = neigh[j][2];
        }
        PCA pca;
        pca.input(neigh_mat);
        pca.compute(PCA::eigen_vector_order::DESCENDING);

        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& eigen_vector = pca.get_eigen_vector();
        if (output_file_name.length() > 0) {
            int dim = _points[i].size();
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& normal = eigen_vector.rightCols(1);
            for (int j = 0; j < dim; ++j) {
                ofs << _points[i][j] << " ";
            }
            for (int j = 0; j < normal.rows(); ++j) {
                ofs << normal(j, 0) << " ";
            }
            ofs << std::endl;
        }
    }
    if (output_file_name.length() > 0) {
        ofs.close();
    }
    return true;
}