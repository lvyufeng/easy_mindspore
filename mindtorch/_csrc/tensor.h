// src/tensor.h
#pragma once

#include <iostream>
#include <vector>
#include <type_traits>
#include <stdexcept>

class Node;

// 定义数据类型
enum DataType {
    SCALAR,
    LIST
};

// 定义Tensor类
template <typename T>
class Tensor;

// 针对标量类型的部分特化
template <typename T>
class Tensor {
private:
    T data;
    DataType dtype;
    size_t size;
    bool requires_grad;
    Tensor* grad;  // 新添加的 grad 属性
    Node* grad_fn;  // 新添加的 grad_fn 属性

public:
    // 构造函数
    explicit Tensor(const T& input, bool requires_grad = false, Node* grad_fn = nullptr);

    // 获取数据类型
    DataType getDataType() const;

    // 获取大小
    size_t getSize() const;

    // 重载加法运算符
    Tensor operator+(const Tensor& other) const;

    // 重载乘法运算符
    Tensor operator*(const Tensor& other) const;

    // 获取数据
    T getData() const;

    // 获取 requires_grad 属性的值
    bool getRequiresGrad() const;

    // 设置 requires_grad 属性的值
    void setRequiresGrad(bool value);
    

    // 获取 grad 属性的值
    Tensor* getGrad() const;

    // 设置 grad 属性的值
    void setGrad(const Tensor& gradient);

    // 获取 grad_fn 属性的值
    Node* getGradFn();

    // 设置 grad_fn 属性的值
    void setGradFn(Node* gradient_function);
};

// 针对std::vector的部分特化
template <typename T>
class Tensor<std::vector<T>> {
private:
    std::vector<T> data;
    DataType dtype;
    size_t size;
    bool requires_grad;
    Tensor* grad;  // 新添加的 grad 属性
    Node* grad_fn;  // 新添加的 grad_fn 属性

public:
    // 构造函数
    explicit Tensor(const std::vector<T>& input, bool requires_grad = false, Node* grad_fn = nullptr);

    // 获取数据类型
    DataType getDataType() const;

    // 获取大小
    size_t getSize() const;

    // 重载加法运算符
    Tensor operator+(const Tensor& other) const;

    // 重载乘法运算符
    Tensor operator*(const Tensor& other) const;

    // 获取数据
    const std::vector<T>& getData() const;

    // 获取 requires_grad 属性的值
    bool getRequiresGrad() const;

    // 设置 requires_grad 属性的值
    void setRequiresGrad(bool value);
    

    // 获取 grad 属性的值
    Tensor* getGrad() const;

    // 设置 grad 属性的值
    void setGrad(const Tensor& gradient);

    // 获取 grad_fn 属性的值
    Node* getGradFn();

    // 设置 grad_fn 属性的值
    void setGradFn(Node* gradient_function);
};

#include "tensor.cpp"
