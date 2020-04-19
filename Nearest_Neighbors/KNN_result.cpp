#include "Nearest_Neighbors/KNN_result.h"

KNNResult::KNNResult(int capacity)
: _capacity(capacity), _count(0), _worst_distance(std::numeric_limits<double>::max()) {
    _distance_value.resize(capacity);
}

bool KNNResult::add_result(double distance, double value) {
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

void KNNResult::print() {
    std::cout << "======= KNNResult =======" << std::endl;
    for (const DistanceValue& dv : _distance_value) {
        std::cout << dv.distance << " : " << dv.value << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

int KNNResult::size() {
    return _count;
}

bool KNNResult::is_full() {
    return _count == _capacity;
}

double KNNResult::worst_distance() {
    return _worst_distance;
}