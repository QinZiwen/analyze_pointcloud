ADD_LIBRARY(Nearest_Neighbors SHARED
    simple_AVLTree.cpp)

ADD_EXECUTABLE(simple_AVLtree_test simple_AVLtree_test.cpp)
TARGET_LINK_LIBRARIES(simple_AVLtree_test Nearest_Neighbors)