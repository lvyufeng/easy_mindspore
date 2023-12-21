// src/Edge.h
#pragma once

class Node;
// 定义Edge类
class Edge {
private:
    Node* node;

public:
    // 构造函数
    Edge(Node* node);

    // 获取节点
    Node* getNode() const;
};

#include "edge.cpp"
