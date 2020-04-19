#include "Nearest_Neighbors/simple_AVLTree.h"

SAVLTree::SAVLTree() : size(0) {}

bool SAVLTree::insert_from_vector(const std::vector<double>& data) {
    for (const double& d : data) {
        insert(d, root);
        size += 1;
    }
    return true;
}

bool SAVLTree::insert(double value, std::shared_ptr<SAVLNode>& root) {
    if (root == nullptr) {
        root.reset(new SAVLNode(value));
    } else {
        if (value < root->value) {
            insert(value, root->left);
            root->LRDiff += 1;
        } else {
            insert(value, root->right);
            root->LRDiff -= 1;
        }

        // std::cout << "root->LRDiff: " << root->LRDiff << std::endl;
        // if (root->left != nullptr) {
        //     std::cout << "root->left->LRDiff" << root->left->LRDiff << std::endl;
        // }
        // if (root->right != nullptr) {
        //     std::cout << "root->right->LRDiff" << root->right->LRDiff << std::endl;
        // }

        // balance
        if (root->LRDiff >= 2 && root->LRDiff * root->left->LRDiff > 0) {  // LL
            rigth_rotate(root);
            root->LRDiff -= 1;
            root->right->LRDiff -= 2;
        } else if (root->LRDiff >= 2 && root->LRDiff * root->left->LRDiff < 0) {  // LR
            left_rotate(root->left);
            root->left->LRDiff += 1;
            root->left->left->LRDiff += 1;
            rigth_rotate(root);
            root->LRDiff -= 1;
            root->right->LRDiff -= 2;
        } else if (root->LRDiff <= -2 && root->LRDiff * root->right->LRDiff > 0) {  // RR
            left_rotate(root);
            root->LRDiff += 1;
            root->left->LRDiff += 2;
        } else if (root->LRDiff <= -2 && root->LRDiff * root->right->LRDiff < 0) {  // RL
            rigth_rotate(root->right);
            root->right->LRDiff -= 1;
            root->right->right->LRDiff -= 1;
            left_rotate(root);
            root->LRDiff += 1;
            root->left->LRDiff += 2;
        }
    }

    return true;
}

bool SAVLTree::rigth_rotate(std::shared_ptr<SAVLNode>& root) {
    std::shared_ptr<SAVLNode> tmp = root->left;
    root->left = tmp->right;
    tmp->right = root;
    root = tmp;
    return true;
}

bool SAVLTree::left_rotate(std::shared_ptr<SAVLNode>& root) {
    std::shared_ptr<SAVLNode> tmp = root->right;
    root->right = tmp->left;
    tmp->left = root;
    root = tmp;
    return true;
}


void SAVLTree::inorder_print(const std::shared_ptr<SAVLNode>& root) {
    if (root != nullptr) {
        inorder_print(root->left);
        std::cout << "value: " << root->value << ", LRDiff: " << root->LRDiff << std::endl;
        inorder_print(root->right);
    }
}

void SAVLTree::preorder_print(const std::shared_ptr<SAVLNode>& root) {
    if (root != nullptr) {
        std::cout << "value: " << root->value << ", LRDiff: " << root->LRDiff << std::endl;
        inorder_print(root->left);
        inorder_print(root->right);
    }
}

void SAVLTree::postorder_print(const std::shared_ptr<SAVLNode>& root) {
    if (root != nullptr) {
        inorder_print(root->left);
        inorder_print(root->right);
        std::cout << "value: " << root->value << ", LRDiff: " << root->LRDiff << std::endl;
    }
}


void SAVLTree::inorder_print() {
    std::cout << "======= inorder_print =======" << std::endl;
    inorder_print(root);
    std::cout << "=============================" << std::endl;
}

void SAVLTree::preorder_print() {
    std::cout << "======= preorder_print =======" << std::endl;
    preorder_print(root);
    std::cout << "==============================" << std::endl;
}

void SAVLTree::postorder_print() {
    std::cout << "======= postorder_print =======" << std::endl;
    postorder_print(root);
    std::cout << "===============================" << std::endl;
}

void SAVLTree::clear() {
    clear(root);
    size = 0;
}

void SAVLTree::clear(std::shared_ptr<SAVLNode>& root) {
    if (root->left == nullptr && root->right == nullptr) {
        root.reset();
    } else {
        if (root->left != nullptr) {
            clear(root->left);
        }
        if (root->right != nullptr) {
            clear(root->right);
        }
        clear(root);
    }
}

size_t SAVLTree::get_size() {
    return size;
}

size_t SAVLTree::get_depth() {
    if (root == nullptr) {
        return 0;
    } else {
        return get_depth(root);
    }
}

size_t SAVLTree::get_depth(const std::shared_ptr<SAVLNode>& root) {
    if (root->left == nullptr && root->right == nullptr) {
        return 1;
    } else {
        size_t l_depth = 0, r_depth = 0;
        if (root->left != nullptr) {
            l_depth = get_depth(root->left);
        }
        if (root->right != nullptr) {
            r_depth = get_depth(root->right);
        }

        return std::max(l_depth, r_depth) + 1;
    }
}

std::shared_ptr<SAVLNode> SAVLTree::get_root() {
    return root;
}