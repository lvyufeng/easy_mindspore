// src/grad_node.h
#pragma once

#include "node.h"
#include "tensor.h"

template <typename T>
class Tensor;

// 定义 AccumulateGrad 类
template <typename T>
class AccumulateGrad : public Node {
private:
    Tensor<T>* leaf_tensor;

public:
    // 构造函数
    AccumulateGrad(Tensor<T>* leaf_tensor);

    // apply 方法的实现
    std::vector<Tensor<T>> apply(Tensor<T>& grad_output);
};

template <typename T>
class AddBackward : public Node {
private:
    T t1_shape;
    T t2_shape;

public:
    // 构造函数
    AddBackward();

    // apply 方法的声明
    std::vector<Tensor<T>> apply(Tensor<T>& grad_output);

    // 设置 t1_shape 的声明
    void set_t1_shape(const T& shape);

    // 设置 t2_shape 的声明
    void set_t2_shape(const T& shape);
};

#include "grad_node.cpp"