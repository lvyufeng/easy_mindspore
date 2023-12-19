// src/functional.h
#pragma once

#include "tensor.h"

// 定义 add 函数
template <typename T>
Tensor<T> add(Tensor<T>& t1, Tensor<T>& t2);

// 辅助函数，用于收集 next_edges
template <typename T, typename... Args>
std::vector<Edge> collect_next_edges(Tensor<T>& t, Args&... tensors);

#include "functional.cpp"