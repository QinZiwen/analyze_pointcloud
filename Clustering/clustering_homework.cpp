#include "Clustering/KMeans.h"
#include "Clustering/GMM.h"
#include "Clustering/spectral.h"
#include "Clustering/mean_shift.h"
#include "Clustering/DBSCAN.h"

#include <sys/time.h>

bool read_data_from_file(const std::string& file_name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& data) {
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    std::string line;
    size_t line_num = 0;
    while(std::getline(ifs, line)) {
        if (line.length() == 0 || line[0] == '#') {
            continue;
        }
        ++line_num;
    }
    ifs.close();

    std::cout << "line_num: " << line_num << std::endl;
    if (line_num <= 0) {
        return false;
    }

    data.resize(2, line_num);
    size_t col_idx = 0;
    ifs.open(file_name, std::ios::in);
    while(std::getline(ifs, line)) {
        if (line.length() == 0 || line[0] == '#') {
            continue;
        }
        std::vector<std::string> line_split = Utils::regexsplit(line, " ");
        if (line_split.size() != 2) {
            std::cout << "file format error, line_split.size(): " << line_split.size() << std::endl;
        }
        data(0, col_idx) = std::stod(line_split[0]);
        data(1, col_idx) = std::stod(line_split[1]);
        ++col_idx;
    }
    ifs.close();
    return true;
}

int main(int argc, char** argv) {
    // 1: k-means; 2: GMM; 3: spectral; 4: mean-shift; 5 DBSCAN
    const int method_type = 5;
    std::string input_path(argv[1]);
    std::string output_path(argv[2]);
    std::vector<std::string> names;
	if (!Utils::get_file_names_from_path(input_path, &names, 8)) {
		std::cerr << "run get_file_names_from_path failure" << std::endl;
		return 1;
	}
    if (!Utils::path_exists(output_path)) {
        if (!Utils::create_path(output_path)) {
            std::cerr << "create path " << output_path << " failure" << std::endl;
            return 1;
        }
    }

    struct timeval process_start;
    struct timeval process_end;
    double process_timer;

    if (method_type == 1) {
        for (const auto& n : names) {
            std::cout << n << std::endl;
            std::string name = input_path + "/" + n;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data;
            if (!read_data_from_file(name, data)) {
                std::cerr << "run read_data_from_file failure" << std::endl;
                return 1;
            }
            // std::cout << "data: " << data << std::endl;

            gettimeofday(&process_start, NULL);
            AAPCD::KMeans kmean;
            kmean.input(data);
            kmean.compute(3);
            gettimeofday(&process_end, NULL);
            process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
            std::cout << "k-means time: " << process_timer << " s" << std::endl;

            std::string output_name = output_path + "/" + n;
            kmean.save_cluster_data_to_file(output_name);
            // break;
        }
    } else if (method_type == 2) {
        for (const auto& n : names) {
            std::cout << n << std::endl;
            std::string name = input_path + "/" + n;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data;
            if (!read_data_from_file(name, data)) {
                std::cerr << "run read_data_from_file failure" << std::endl;
                return 1;
            }

            gettimeofday(&process_start, NULL);
            AAPCD::GMM gmm;
            gmm.input(data);
            gmm.compute(3, 100, 0.001);
            gettimeofday(&process_end, NULL);
            process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
            std::cout << "GMM time: " << process_timer << " s" << std::endl;

            std::string output_name = output_path + "/" + n;
            gmm.save_cluster_data_to_file(output_name);
            // break;
        }
    } else if (method_type == 3) {
        for (const auto& n : names) {
            std::cout << n << std::endl;
            std::string name = input_path + "/" + n;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data;
            if (!read_data_from_file(name, data)) {
                std::cerr << "run read_data_from_file failure" << std::endl;
                return 1;
            }
            // std::cout << "data:\n" << data << std::endl;

            gettimeofday(&process_start, NULL);
            AAPCD::Spectral spectral;
            spectral.input(data);
            spectral.compute(3, 100, 0.001, 
                AAPCD::Spectral::ADJACENCY_METHOD::FULL_CONNECT,
                AAPCD::Spectral::NORMALIZED_LAPLACIAN::RW);
            gettimeofday(&process_end, NULL);
            process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
            std::cout << "Spectral time: " << process_timer << " s" << std::endl;

            std::string output_name = output_path + "/" + n;
            spectral.save_cluster_data_to_file(output_name);
            // break;
        }
    }  else if (method_type == 4) {
        for (const auto& n : names) {
            std::cout << n << std::endl;
            std::string name = input_path + "/" + n;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data;
            if (!read_data_from_file(name, data)) {
                std::cerr << "run read_data_from_file failure" << std::endl;
                return 1;
            }
            // std::cout << "data:\n" << data << std::endl;

            gettimeofday(&process_start, NULL);
            AAPCD::MeanShift mean_shift;
            mean_shift.input(data);
            mean_shift.compute(0.6, 0.001, AAPCD::MeanShift::ClusterMethod::KMEANS);
            gettimeofday(&process_end, NULL);
            process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
            std::cout << "Spectral time: " << process_timer << " s" << std::endl;

            std::string output_name = output_path + "/" + n;
            mean_shift.save_cluster_data_to_file(output_name);
            // break;
        }
    } if (method_type == 5) {
        for (const auto& n : names) {
            std::cout << n << std::endl;
            std::string name = input_path + "/" + n;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data;
            if (!read_data_from_file(name, data)) {
                std::cerr << "run read_data_from_file failure" << std::endl;
                return 1;
            }
            // std::cout << "data:\n" << data << std::endl;

            gettimeofday(&process_start, NULL);
            AAPCD::DBSCAN dbscan;
            dbscan.input(data);
            dbscan.compute(0.45, 5);
            gettimeofday(&process_end, NULL);
            process_timer = process_end.tv_sec - process_start.tv_sec + (float)(process_end.tv_usec - process_start.tv_usec)/1000000; 
            std::cout << "Spectral time: " << process_timer << " s" << std::endl;
            dbscan.print();

            std::string output_name = output_path + "/" + n;
            dbscan.save_cluster_data_to_file(output_name);
            // break;
        }
    } else {
        std::cerr << "method type invalid: " << method_type << std::endl;
    }

    return 0;
}