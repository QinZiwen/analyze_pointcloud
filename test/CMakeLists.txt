SET(LIBS PCA KD_tree)

add_executable(kd_tree_test kd_tree_test.cpp)
target_link_libraries(kd_tree_test ${LIBS})