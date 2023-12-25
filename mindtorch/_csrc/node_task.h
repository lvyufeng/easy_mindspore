
#pragma once

#include "tensor.h"
#include "node.h"

template <typename T>
class NodeTask {
public:
    NodeTask(Node* node, Tensor<T> grad_input);
    void update_grad_input(const Tensor<T> grad_input);

    Node* node;
    Tensor<T> grad_input;
};
