// src/grad_node.cpp
#include "grad_node.h"

// 实现 AccumulateGrad 类的方法

template <typename T>
AccumulateGrad<T>::AccumulateGrad(Tensor<T>* leaf_tensor) : leaf_tensor(leaf_tensor) {}

template <typename T>
std::vector<Tensor<T>> AccumulateGrad<T>::apply(Tensor<T>& grad_output) {
    if (leaf_tensor->getGrad() == nullptr) {
        leaf_tensor->setGrad(grad_output);
    } else {
        leaf_tensor->setGrad(*(leaf_tensor->getGrad()) + grad_output);
    }
    // Return an empty Tensor, as AccumulateGrad does not produce output
    return {};
}

template <typename T>
AddBackward<T>::AddBackward() : t1_shape(0), t2_shape(0) {}

// apply 方法的实现
template <typename T>
std::vector<Tensor<T>> AddBackward<T>::apply(Tensor<T>& grad_output) {
    std::vector<Tensor<T>> grad_input;
    if (t1_shape != 0) {
        Tensor<T> x = grad_output;
        grad_input.push_back(x);
    }
    if (t2_shape != 0) {
        Tensor<T> y = grad_output;
        grad_input.push_back(y);
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

