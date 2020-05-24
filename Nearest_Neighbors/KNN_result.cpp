#include "Nearest_Neighbors/KNN_result.h"

KNNResultNumber::KNNResultNumber(int capacity)
    : _capacity(capacity), _count(0), _worst_distance(std::numeric_limits<double>::max()) {
    _distance_value.resize(capacity);
}

bool KNNResultNumber::add_result(double distance, double value) {
    if (distance > _worst_distance) {
        return false;
    }

    if (_count < _capacity) {
        ++_count;
    }

    // If a value is added, put it in a ordered position
    int index = _count - 1;
    while (index > 0) {
        if (_distance_value[index - 1].distance > distance) {
            _distance_value[index] = _distance_value[index - 1];
            --index;
        } else {
            break;
        }
    }

    _distance_value[index].distance = distance;
    _distance_value[index].value = value;
    _worst_distance = _distance_value[_capacity - 1].distance;
    return true;
}

void KNNResultNumber::print() {
    std::cout << "======= KNNResultNumber =======" << std::endl;
    for (const DistanceValue& dv : _distance_value) {
        std::cout << dv.distance << " : " << dv.value << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

std::vector<DistanceValue> KNNResultNumber::get_distance_value() {
    return _distance_value;
}

int KNNResultNumber::size() {
    return _count;
}

bool KNNResultNumber::is_full() {
    return _count == _capacity;
}

double KNNResultNumber::worst_distance() {
    return _worst_distance;
}

KNNResultRadius::KNNResultRadius(double rad)
    : _radius(rad) {
    _distance_value.clear();
}

bool KNNResultRadius::add_result(double distance, double value) {
    if (distance > _radius) {
        return false;
    }

    _distance_value.emplace_back(DistanceValue(distance, value));
    return true;
}

int KNNResultRadius::size() {
    return _distance_value.size();
}

void KNNResultRadius::print() {
    std::cout << "======= KNNResultRadius =======" << std::endl;
    for (const DistanceValue& dv : _distance_value) {
        std::cout << dv.distance << " : " << dv.value << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

double KNNResultRadius::worst_distance() {
    return _radius;
}

std::vector<DistanceValue> KNNResultRadius::get_distance_value() {
    return _distance_value;
}