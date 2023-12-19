// src/Tensor.cpp
#include "tensor.h"
#include "node.h"

// 针对标量类型的实现
template <typename T>
Tensor<T>::Tensor(const T& input, bool requires_grad, Node* grad_fn)
     : data(input), dtype(SCALAR), size(1) , requires_grad(requires_grad), grad(nullptr), grad_fn(grad_fn){
    // 判断数据类型
    if (!std::is_scalar<T>::value) {
        throw std::invalid_argument("Invalid data type for Tensor");
    }
}

template <typename T>
DataType Tensor<T>::getDataType() const {
    return dtype;
}

template <typename T>
size_t Tensor<T>::getSize() const {
    return size;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    if (dtype == SCALAR && other.getDataType() == SCALAR) {
        return Tensor(data + other.getData());
    } else {
        throw std::invalid_argument("Incompatible data types for addition");
    }
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
    if (dtype == SCALAR && other.getDataType() == SCALAR) {
        return Tensor(data * other.getData());
    } else {
        throw std::invalid_argument("Incompatible data types for multiplication");
    }
}

template <typename T>
T Tensor<T>::getData() const {
    return data;
}

// 针对std::vector的实现
template <typename T>
Tensor<std::vector<T>>::Tensor(const std::vector<T>& input, bool requires_grad, Node* grad_fn)
    : data(input), dtype(LIST), size(data.size()), requires_grad(requires_grad), grad(nullptr), grad_fn(grad_fn) {}

template <typename T>
DataType Tensor<std::vector<T>>::getDataType() const {
    return dtype;
}

template <typename T>
size_t Tensor<std::vector<T>>::getSize() const {
    return size;
}

template <typename T>
Tensor<std::vector<T>> Tensor<std::vector<T>>::operator+(const Tensor& other) const {
    if (dtype == LIST && other.getDataType() == LIST) {
        if (size == other.getSize()) {
            // 对应位置元素相加
            std::vector<T> result;
            result.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result.push_back(data[i] + other.getData()[i]);
            }
            return Tensor(result);
        } else {
            throw std::invalid_argument("Incompatible sizes for list addition");
        }
    } else {
        throw std::invalid_argument("Incompatible data types for addition");
    }
}

template <typename T>
Tensor<std::vector<T>> Tensor<std::vector<T>>::operator*(const Tensor& other) const {
    if (dtype == LIST && other.getDataType() == LIST) {
        if (size == other.getSize()) {
            // 对应位置元素相乘
            std::vector<T> result;
            result.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result.push_back(data[i] * other.getData()[i]);
            }
            return Tensor(result);
        } else {
            throw std::invalid_argument("Incompatible sizes for list multiplication");
        }
    } else {
        throw std::invalid_argument("Incompatible data types for multiplication");
    }
}

template <typename T>
const std::vector<T>& Tensor<std::vector<T>>::getData() const {
    return data;
}

template <typename T>
bool Tensor<T>::getRequiresGrad() const {
    return requires_grad;
}

template <typename T>
void Tensor<T>::setRequiresGrad(bool value) {
    requires_grad = value;
}

template <typename T>
Tensor<T>* Tensor<T>::getGrad() const {
    return grad;
}

template <typename T>
void Tensor<T>::setGrad(const Tensor& gradient) {
    grad = new Tensor(gradient);
}

template <typename T>
Node* Tensor<T>::getGradFn() {
    return grad_fn;
}

template <typename T>
void Tensor<T>::setGradFn(Node* gradient_function) {
    grad_fn = gradient_function;
}


template class Tensor<int>;
template class Tensor<std::vector<int>>;