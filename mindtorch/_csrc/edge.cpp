// src/Edge.cpp
#include "edge.h"

// 实现Edge类的方法

Edge::Edge(Node* node) : node(node) {}

Node* Edge::get_node() const {
    return node;
}
