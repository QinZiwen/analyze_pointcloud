#include <iostream>
#include <string>
#include <fstream>

#include "Fitting/find_ground.h"

bool read_and_save_points(const std::string& file_name, const std::string& output_name) {
    std::ifstream ifs(file_name, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }
    std::ofstream ofs(output_name, std::ios::out);
    if (!ofs.is_open()) {
        std::cerr << "can not open " << output_name << std::endl;
        return false;
    }

    const int size = 4;
    float fea[size];
    int n=0;
    while(ifs.read((char *)&fea[0], size*sizeof(float))){
        ofs << fea[0] << " " << fea[1] << " " << fea[2] << std::endl;
        ++n;
    }

    ofs.close();
    ifs.close();
    return true;
}

bool read_data_as_matrix(const std::string& file_name, Eigen::MatrixXd& data) {
    std::ifstream ifs(file_name, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "can not open " << file_name << std::endl;
        return false;
    }

    // calc pointcloud size
    ifs.seekg(0, ifs.end);
    auto end = ifs.tellg();
    ifs.seekg(0, ifs.beg);

    auto size = end * sizeof(char) / sizeof(float);
    auto data_size = size / 4;
    data.resize(3, data_size);

    const int bsize = 4;
    float fea[bsize];
    int n=0;
    while(ifs.read((char *)&fea[0], bsize*sizeof(float))){
        data(0, n) = fea[0];
        data(1, n) = fea[1];
        data(2, n) = fea[2];
        ++n;
    }

    ifs.close();
    return true;
}

int main(int argc, char** argv) {
    std::string input_name(argv[1]);
    Eigen::MatrixXd data;
    if (!read_data_as_matrix(input_name, data)) {
        std::cerr << "read_data_as_matrix failed" << std::endl;
        return -1;
    }
    std::cout << "data size: " << data.cols() << std::endl;

    AAPCD::FindGround find_ground;
    find_ground.input(data);
    find_ground.find();
    return 0;
}

// int main(int argc, char** argv) {
//     std::string input_name(argv[1]);
//     std::string output_name(argv[2]);
//     if (!read_and_save_points(input_name, output_name)) {
//         std::cerr << "read_and_save_points failed" << std::endl;
//         return -1;
//     }
//     return 0;
// }