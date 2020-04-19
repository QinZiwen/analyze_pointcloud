#pragma once

#include "utils/utils.hpp"

#include <memory>

typedef struct SAVLNode {
public:
    double value;
    int LRDiff;
    std::shared_ptr<struct SAVLNode> left;
    std::shared_ptr<struct SAVLNode> right;

    SAVLNode(double value)
    : value(value), LRDiff(0), left(nullptr), right(nullptr)
    {}
} SAVLNode;

class SAVLTree {
public:
    SAVLTree();
    bool insert_from_vector(const std::vector<double>& data);

    void clear();

    void inorder_print();
    void preorder_print();
    void postorder_print();
    
    size_t get_size();
    size_t get_depth();
    std::shared_ptr<SAVLNode> get_root();

private:
    bool insert(double value, std::shared_ptr<SAVLNode>& root);

    bool rigth_rotate(std::shared_ptr<SAVLNode>& root);
    bool left_rotate(std::shared_ptr<SAVLNode>& root);
    
    void inorder_print(const std::shared_ptr<SAVLNode>& root);
    void preorder_print(const std::shared_ptr<SAVLNode>& root);
    void postorder_print(const std::shared_ptr<SAVLNode>& root);
    void clear(std::shared_ptr<SAVLNode>& root);

    size_t get_depth(const std::shared_ptr<SAVLNode>& root);

private:
    std::shared_ptr<SAVLNode> root;
    size_t size;
};