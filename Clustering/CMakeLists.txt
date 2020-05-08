ADD_LIBRARY(Clustering SHARED
    KMeans.cpp)

ADD_EXECUTABLE(KMeans_test KMeans_test.cpp)
TARGET_LINK_LIBRARIES(KMeans_test Clustering)

ADD_EXECUTABLE(clustering_homework clustering_homework.cpp)
TARGET_LINK_LIBRARIES(clustering_homework Clustering)