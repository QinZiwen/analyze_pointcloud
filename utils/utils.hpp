#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

#include <Eigen/Dense>

class Utils {
public:
static unsigned long long  get_now_timestamp() {
  // get timestamp
	using namespace std::chrono;
  milliseconds ms = duration_cast< milliseconds >(
      system_clock::now().time_since_epoch()
  );
	unsigned long long timestamp = ms.count();
  return timestamp;
}

static std::vector<std::string> regexsplit(const std::string& input, const std::string& reg) {
  std::string s(input);
  std::vector<std::string> vec;
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(reg)) != std::string::npos) {
    if (pos == 0) {
      s.erase(0, pos + reg.length());
      continue;
    }
    vec.emplace_back(s.substr(0, pos));
    s.erase(0, pos + reg.length());
  }
  if (s.length() > 0) {
    vec.emplace_back(s);
  }
  return vec;
}

static bool get_file_name_from_path(const std::string& path, std::string* file_name) {
  if (path.empty() == true) {
      std::cerr << "get_file_name_from_path input path is empty" << std::endl;
      return false;
  }

  std::string tmp_path = path;
  if (path[path.length() - 1] == '/') {
      tmp_path = path.substr(0, path.length() - 1);
  }

  auto pos = tmp_path.find_last_of('/');
  if (pos != std::string::npos) {
      *file_name = tmp_path.substr(pos + 1, tmp_path.length());
  } else {
      *file_name = tmp_path;
  }
  return true;
}

static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> read_pointcloud_from_file(const std::string& file_name) {
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
        std::cerr << "[FATAL ERROR] can not open " << file_name << std::endl;
        exit(1);
    }

    int pcd_size = 0;
    std::string line;
    // get number of points
    while(std::getline(ifs, line)) {
        if (line.length() == 0 || line[0] == '#') {
            continue;
        }
        ++pcd_size;
    }
    ifs.close();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> res(3, pcd_size);
    int index = 0;
    ifs.open(file_name, std::ios::in);
    while(std::getline(ifs, line)) {
        if (line.length() == 0 || line[0] == '#') {
            continue;
        }
        std::vector<std::string> line_splits = regexsplit(line, ",");
        if (line_splits.size() < 3) {
            std::cerr << "[FATAL ERROR] input file format error" << std::endl;
            exit(1);
        }

        res(0, index) = std::stod(line_splits[0]);
        res(1, index) = std::stod(line_splits[1]);
        res(2, index) = std::stod(line_splits[2]);
        ++index;
    }
    ifs.close();
    return res;
}

static bool save_pointcloud_to_file(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& pcd, const std::string& file_name) {
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
        std::cerr << "[FATAL ERROR] can not open " << file_name << std::endl;
        return false;
    }

    for (size_t i = 0; i < pcd.rows(); ++i) {
        if (pcd.cols() == 1) {
            ofs << pcd(i, 0) << " 0 0";
        } else if (pcd.cols() == 2) {
            for (size_t j = 0; j < 2; ++j) {
                ofs << pcd(i, j) << " ";
            }
            ofs << "0";
        } else if (pcd.cols() == 3) {
            for (size_t j = 0; j < pcd.cols(); ++j) {
                ofs << pcd(i, j) << " ";
            }
        } else {
            std::cerr << "input data format error, cols: " << pcd.cols() << std::endl;
            return false;
        }
        
        ofs << std::endl;
    }

    ofs.close();
    return true;
}
};