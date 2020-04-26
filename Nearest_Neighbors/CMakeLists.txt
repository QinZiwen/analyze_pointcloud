ADD_LIBRARY(Nearest_Neighbors SHARED
    simple_AVLTree.cpp
    KNN_result.cpp
    nearest_neighbors.cpp
    KD_Tree.cpp
    Octree.cpp)

ADD_EXECUTABLE(simple_AVLtree_test simple_AVLtree_test.cpp)
TARGET_LINK_LIBRARIES(simple_AVLtree_test Nearest_Neighbors)

ADD_EXECUTABLE(KNN_result_test KNN_result_test.cpp)
TARGET_LINK_LIBRARIES(KNN_result_test Nearest_Neighbors)

ADD_EXECUTABLE(nearest_neighbors_test nearest_neighbors_test.cpp)
TARGET_LINK_LIBRARIES(nearest_neighbors_test Nearest_Neighbors)

ADD_EXECUTABLE(KD_Tree_test KD_Tree_test.cpp)
TARGET_LINK_LIBRARIES(KD_Tree_test Nearest_Neighbors)

ADD_EXECUTABLE(Octree_test Octree_test.cpp)
TARGET_LINK_LIBRARIES(Octree_test Nearest_Neighbors)

ADD_EXECUTABLE(nearest_neighbors_homework nearest_neighbors_homework.cpp)
TARGET_LINK_LIBRARIES(nearest_neighbors_homework Nearest_Neighbors KD_tree)