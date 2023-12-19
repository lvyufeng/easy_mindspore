// src/Node.h
#pragma once

#include <vector>
#include "tensor.h"

// 前向声明
class Edge;

// 定义Node类
class Node {
private:
    std::vector<Edge> next_edges;

public:
    // 构造函数
    Node();

    // 设置 next_edges
    void set_next_edges(const std::vector<Edge>& edges);

    // 应用操作
    virtual std::vector<Tensor<int>> apply(const std::vector<Tensor<int>>& inputs);
};

// apply 函数的虚函数实现（在派生类中实现）
std::vector<Tensor<int>> Node::apply(const std::vector<Tensor<int>>& inputs) {
    // 在派生类中实现具体的操作
    // 返回结果作为 Tensor<int> 的向量
    return std::vector<Tensor<int>>{};
}

#include "node.cpp"
