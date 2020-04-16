#include "PCA_Downsample/normal_estimator.h"

int main(int argc, char** argv) {
    std::string pts_file(argv[1]);
    std::string output_file(argv[2]);
    NormalEstimator ne;
    ne.set_data(pts_file);
    ne.compute(0.1, output_file);

    return 0;
}