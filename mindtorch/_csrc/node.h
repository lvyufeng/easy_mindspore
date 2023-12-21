// src/Node.h
#pragma once

#include <vector>
#include "tensor.h"

// 前向声明
class Edge;

// 定义Node类
class Node {
private:
    static int next_id;  // 静态变量，用于生成唯一的 ID
    int id;              // 每个 Node 实例的唯一 ID
    std::vector<Edge> next_edges;

public:
    // 构造函数
    Node();

    // 设置 next_edges
    void set_next_edges(const std::vector<Edge>& edges);
    const std::vector<Edge>& getNextEdges() const;

    // 获取 Node 的唯一 ID
    int getId() const;

    bool hasNextEdges() const;

    virtual std::string getName() const;

    // 应用操作
    virtual std::vector<Tensor<int>> apply(const std::vector<Tensor<int>>& inputs);
    virtual std::vector<Tensor<int>> apply(Tensor<int>& inputs);
};

// apply 函数的虚函数实现（在派生类中实现）
std::vector<Tensor<int>> Node::apply(const std::vector<Tensor<int>>& inputs) {
    // 在派生类中实现具体的操作
    // 返回结果作为 Tensor<int> 的向量
    return std::vector<Tensor<int>>{};
}

std::vector<Tensor<int>> Node::apply(Tensor<int>& inputs) {
    // 在派生类中实现具体的操作
    // 返回结果作为 Tensor<int> 的向量
    return std::vector<Tensor<int>>{};
}

#include "node.cpp"
