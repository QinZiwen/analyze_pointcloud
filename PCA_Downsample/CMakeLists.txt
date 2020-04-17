ADD_LIBRARY(PCA SHARED
    pca.cpp
    normal_estimator.cpp)
TARGET_LINK_LIBRARIES(PCA KD_tree)

ADD_LIBRARY(voxel_grid_downsampling SHARED
    voxel_grid_downsampling.cpp)

ADD_EXECUTABLE(pca_test pca_test.cpp)
TARGET_LINK_LIBRARIES(pca_test PCA)

ADD_EXECUTABLE(pcd_dimension_reduction pcd_dimension_reduction.cpp)
TARGET_LINK_LIBRARIES(pcd_dimension_reduction PCA)

ADD_EXECUTABLE(estimate_pts_normal estimate_pts_normal.cpp)
TARGET_LINK_LIBRARIES(estimate_pts_normal PCA)

ADD_EXECUTABLE(pcd_downsample pcd_downsample.cpp)
TARGET_LINK_LIBRARIES(pcd_downsample voxel_grid_downsampling)