ADD_LIBRARY(PCA SHARED
    pca.cpp)

add_executable(pca_test pca_test.cpp)
target_link_libraries(pca_test PCA)

add_executable(pcd_dimension_reduction pcd_dimension_reduction.cpp)
target_link_libraries(pcd_dimension_reduction PCA)