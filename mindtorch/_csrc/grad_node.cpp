// src/grad_node.cpp
#include "grad_node.h"

// 实现 AccumulateGrad 类的方法

template <typename T>
AccumulateGrad<T>::AccumulateGrad(Tensor<T>* leaf_tensor) : leaf_tensor(leaf_tensor) {}

template <typename T>
std::vector<Tensor<T>> AccumulateGrad<T>::apply(std::vector<Tensor<T>>& grad_output) {
    if (leaf_tensor->getGrad() == nullptr) {
        leaf_tensor->setGrad(grad_output[0]);
    } else {
        leaf_tensor->setGrad(leaf_tensor->getGrad() + grad_output[0]);
    }
    // Return an empty Tensor, as AccumulateGrad does not produce output
    return {};
}

template <typename T>
AddBackward<T>::AddBackward() : t1_shape(), t2_shape() {}

// apply 方法的实现
template <typename T>
std::vector<T> AddBackward<T>::apply(const Tensor<T>& grad_output) {
    std::vector<T> grad_input;
    if (t1_shape != T()) {
        grad_input.push_back(grad_output);
    }
    if (t2_shape != T()) {
        grad_input.push_back(grad_output);
    }
    return grad_input;
}

// 设置 t1_shape 的实现
template <typename T>
void AddBackward<T>::set_t1_shape(const T& shape) {
    t1_shape = shape;
}

// 设置 t2_shape 的实现
template <typename T>
void AddBackward<T>::set_t2_shape(const T& shape) {
    t2_shape = shape;
}

