ADD_LIBRARY(PCA SHARED
    pca.cpp)

add_executable(pca_test pca_test.cpp)
target_link_libraries(pca_test PCA)
