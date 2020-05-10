ADD_LIBRARY(Clustering SHARED
    KMeans.cpp
    GMM.cpp
    spectral.cpp)

ADD_EXECUTABLE(KMeans_test KMeans_test.cpp)
TARGET_LINK_LIBRARIES(KMeans_test Clustering)

ADD_EXECUTABLE(clustering_homework clustering_homework.cpp)
TARGET_LINK_LIBRARIES(clustering_homework Clustering)

ADD_EXECUTABLE(GMM_test GMM_test.cpp)
TARGET_LINK_LIBRARIES(GMM_test Clustering)