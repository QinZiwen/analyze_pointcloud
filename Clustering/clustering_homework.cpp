#include "Clustering/KMeans.h"

int main(int argc, char** argv) {
    std::string input_path(argv[1]);
    std::vector<std::string> names;
    if (!Utils::get_absolute_file_name_frome_path(input_path, names)) {
        std::cerr << "run get_absolute_file_name_frome_path failure" << std::endl;
        return 1;
    }
    for (const auto& n : names) {
        std::cout << n << std::endl;
    }

    return 0;
}