#include "KD_tree/KD_tree.hpp"

#include <iostream>
#include <vector>

int main() {
    pointVec points;
    point_t pt;

    pt = {0.0, 0.0};
    points.push_back(pt);
    pt = {1.0, 0.0};
    points.push_back(pt);
    pt = {0.0, 1.0};
    points.push_back(pt);
    pt = {1.0, 1.0};
    points.push_back(pt);
    pt = {0.5, 0.5};
    points.push_back(pt);

    KDTree tree(points);

    std::cout << "nearest test\n";
    pt = {0.8, 0.2};
    auto res = tree.nearest_point(pt);
    for (double b : res) {
        std::cout << b << " ";
    }
    std::cout << '\n';
 
    auto res2 = tree.neighborhood_points(pt, .55);

    for (point_t a : res2) {
        for (double b : a) {
            std::cout << b << " ";
        }
        std::cout << '\n';
    }
    return 0;
}