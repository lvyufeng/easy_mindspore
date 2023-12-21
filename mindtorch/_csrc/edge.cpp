// src/Edge.cpp
#include "edge.h"

// 实现Edge类的方法

Edge::Edge(Node* node) : node(node) {}

Node* Edge::getNode() const {
    return node;
}
