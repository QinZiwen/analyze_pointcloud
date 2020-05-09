#include "Clustering/KMeans.h"

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

    for (const auto& n : names) {
        std::cout << n << std::endl;
        std::string name = input_path + "/" + n;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data;
        if (!read_data_from_file(name, data)) {
            std::cerr << "run read_data_from_file failure" << std::endl;
            return 1;
        }
        std::cout << "data: " << data << std::endl;

        AAPCD::KMeans kmean;
        kmean.input(data);
        kmean.compute(2);

        std::string output_name = output_path + "/" + n;
        kmean.save_cluster_data_to_file(output_name);
        // break;
    }

    return 0;
}