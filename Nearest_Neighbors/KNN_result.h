#pragma once

#include "utils/utils.hpp"

#include <limits> 

typedef struct DistanceValue {
public:
    double distance;
    double value;

    DistanceValue()
    : distance(std::numeric_limits<double>::max()), value(0)
    {}

    DistanceValue(double dis, double v)
    : distance(dis), value(v)
    {}
} DistanceValue;

class KNNResult {
public:
    virtual bool add_result(double distance, double value) = 0;
};

class KNNResultNumber : public KNNResult{
public:
    KNNResultNumber(int capacity);

    bool add_result(double distance, double value);

    int size();
    bool is_full();
    double worst_distance();
    void print();
    std::vector<DistanceValue> get_distance_value();

private:
    int _capacity;
    int _count;
    double _worst_distance;
    std::vector<DistanceValue> _distance_value;
};

class KNNResultRadius : public KNNResult {
public:
    KNNResultRadius(double rad);
    bool add_result(double distance, double value);

    int size();
    void print();
    double worst_distance();
    std::vector<DistanceValue> get_distance_value();

private:
    double _radius;
    std::vector<DistanceValue> _distance_value;
};
