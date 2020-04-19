#include "Nearest_Neighbors/simple_AVLTree.h"

int main(int argc, char** argv) {
    std::vector<double> data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> data2 = {1, 3, 2, 4, 5, 6, 7, 8, 9};
    std::vector<double> data3 = {3, 2, 1, 4, 5, 6, 7, 8, 9};
    std::vector<double> data4 = {3, 1, 2, 4, 5, 6, 7, 8, 9};

    SAVLTree savl_tree;
    savl_tree.insert_from_vector(data1);
    savl_tree.preorder_print();
    std::cout << "size: " << savl_tree.get_size() << std::endl;
    std::cout << "depth: " << savl_tree.get_depth() << std::endl;
    savl_tree.clear();

    savl_tree.insert_from_vector(data2);
    savl_tree.preorder_print();
    std::cout << "size: " << savl_tree.get_size() << std::endl;
    std::cout << "depth: " << savl_tree.get_depth() << std::endl;
    savl_tree.clear();

    savl_tree.insert_from_vector(data3);
    savl_tree.preorder_print();
    std::cout << "size: " << savl_tree.get_size() << std::endl;
    std::cout << "depth: " << savl_tree.get_depth() << std::endl;
    savl_tree.clear();

    savl_tree.insert_from_vector(data4);
    savl_tree.preorder_print();
    std::cout << "size: " << savl_tree.get_size() << std::endl;
    std::cout << "depth: " << savl_tree.get_depth() << std::endl;
    savl_tree.clear();

    // ------------------------------------
    // std::vector<double> data1 = {1, 2, 3};
    // std::vector<double> data2 = {1, 3, 2};
    // std::vector<double> data3 = {3, 2, 1};
    // std::vector<double> data4 = {3, 1, 2};

    // SAVLTree savl_tree;
    // savl_tree.insert_from_vector(data1);
    // savl_tree.preorder_print();
    // savl_tree.clear();

    // savl_tree.insert_from_vector(data2);
    // savl_tree.preorder_print();
    // savl_tree.clear();

    // savl_tree.insert_from_vector(data3);
    // savl_tree.preorder_print();
    // savl_tree.clear();

    // savl_tree.insert_from_vector(data4);
    // savl_tree.preorder_print();
    // savl_tree.clear();
}