#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

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

/**
* Get all file names under the specified path
* type: if type == 8, the function will return all file names; if type == 4, the function will return all directory names
*/
static bool get_file_names_from_path(const std::string& path, std::vector<std::string>* file_name, const int& type) {
    if (file_name == nullptr) {
        std::cerr << "file_name is nullptr" << std::endl;
        return false;
    }

    DIR *dp;
    struct dirent *dirp;

    if ((dp = opendir(path.c_str())) == nullptr) {
        std::cerr << "Can not open " << path << std::endl;
        return false;
    }

    while ((dirp = readdir(dp)) != nullptr) {
        if (!strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, "..")) {
            continue;
        }

        if (type == static_cast<int>(dirp->d_type)) {
            file_name->emplace_back(dirp->d_name);
        }
    }
    closedir(dp);

    return true;
}

static bool get_absolute_file_name_frome_path(const std::string& path, std::vector<std::string>& names) {
	std::vector<std::string> file_name;
	if (!get_file_names_from_path(path, &file_name, 8)) {
		std::cerr << "run get_file_names_from_path failure" << std::endl;
		return false;
	}

	for (const auto& n : file_name) {
		names.emplace_back(path + "/" + n);
	}
    return true;
}

static bool run_command(const std::string& command) {
	if (command.empty()) {
		std::cerr << "empty command!" << std::endl;
		return false;
	}
	int com_res = std::system(command.c_str());
	std::cout << "run " + command + " status: " << com_res << std::endl;
	if (com_res != 0) {
		std::cerr << "run " + command + " failed" << std::endl;
		return false;
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

static bool read_point_cloud_from_bin(
    const std::string bin_file,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& pcd) {
    std::ifstream point_stream;
    point_stream.open(bin_file.c_str(), std::ios::binary);
    if (!point_stream.good()) {
        std::cerr << "[FATAL ERROR] can not open " << bin_file << std::endl;
        return false;
    }

    point_stream.seekg(0, point_stream.end);
    int64_t current_index = point_stream.tellg();
    point_stream.seekg(0, std::ios::beg);

    int64_t size = current_index * sizeof(char) / sizeof(float);
    float* buffer = new float[size];
    point_stream.read(reinterpret_cast<char*>(buffer), current_index);

    pcd.resize(3, size / 3);
    for (int i = 0; i < size; i += 3) {
        pcd(0, i / 3) = double(buffer[i]);      // x
        pcd(1, i / 3) = double(buffer[i + 1]);  // y
        pcd(2, i / 3) = double(buffer[i + 2]);  // z
    }
    delete[] buffer;
    point_stream.close();

    std::cout << "read " << pcd.cols() << " points" << std::endl;
    return true;
}

static bool path_exists(const std::string& path) {
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFDIR)) {
    return true;
  }
  return false;
}

static bool create_path(const std::string& path_to_create_input) {
  constexpr mode_t kMode = 0777;

  if (path_to_create_input.empty()) {
    std::cout << "\033[31mCannot create empty path!\033[0m";
    return false;
  } 

  // Append slash if necessary to make sure that stepping through the folders
  // works.
  std::string path_to_create = path_to_create_input;
  if (path_to_create.back() != '/') {
    path_to_create += '/';
  }

  // Loop over the path and create one folder after another.
  size_t current_position = 0u;
  size_t previous_position = 0u;
  std::string current_directory;
  while ((current_position = path_to_create.find_first_of(
              '/', previous_position)) != std::string::npos) {
    current_directory = path_to_create.substr(0, current_position++);
    previous_position = current_position;

    if (current_directory == "." || current_directory.empty()) {
      continue;
    }

    int make_dir_status = 0;
    if ((make_dir_status = mkdir(current_directory.c_str(), kMode)) &&
        errno != EEXIST) {
      std::cout << "\033[31mUnable to make path! Error: " << strerror(errno) << "\033[0m";
      return make_dir_status == 0;
    }
  }
  return true;
}

};