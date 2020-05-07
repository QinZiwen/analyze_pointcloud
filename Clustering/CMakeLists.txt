ADD_LIBRARY(Clustering SHARED
    KMeans.cpp)

ADD_EXECUTABLE(KMeans_test KMeans_test.cpp)
TARGET_LINK_LIBRARIES(KMeans_test Clustering)