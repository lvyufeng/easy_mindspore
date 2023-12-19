// src/Node.cpp
#include "node.h"

// 实现Node类的方法
class Edge;

Node::Node() {}

void Node::set_next_edges(const std::vector<Edge>& edges) {
    next_edges = edges;
}