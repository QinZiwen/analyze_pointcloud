#include "PCA_Downsample/pca.h"

int main(int argc, char** argv) {
    Eigen::Matrix<double, 2, 3> data;
    data << 1, 2, 3,
            1, 2, 3;
    
    PCA pca;
    pca.input(data);
    pca.compute(PCA::eigen_vector_order::DESCENDING);
    std::cout << "eigen values: \n" << pca.get_eigen_values() << std::endl;
    std::cout << "eigen vector: \n" << pca.get_eigen_vector() << std::endl;
    return 0;
}