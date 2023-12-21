// src/Node.cpp
#include "node.h"

// 实现Node类的方法
class Edge;

int Node::next_id = 0;  // 初始化静态变量

Node::Node() {
    id = next_id++;
}

int Node::getId() const {
    return id;
}

std::string Node::getName() const {
    return typeid(*this).name();
}


void Node::set_next_edges(const std::vector<Edge>& edges) {
    next_edges = edges;
}

const std::vector<Edge>& Node::getNextEdges() const {
    return next_edges;
}

bool Node::hasNextEdges() const {
    return !next_edges.empty();
}