ADD_LIBRARY(Fitting SHARED
    find_ground.cpp)
TARGET_LINK_LIBRARIES(Fitting
    Nearest_Neighbors
    PCA Clustering
    voxel_grid_downsampling)

ADD_EXECUTABLE(find_ground_test find_ground_test.cpp)
TARGET_LINK_LIBRARIES(find_ground_test Fitting)