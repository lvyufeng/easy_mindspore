// NodeTask.cpp

#include "node_task.h"

template class NodeTask<int>;

template <typename T>
NodeTask<T>::NodeTask(Node* node, Tensor<T> grad_input)
    : node(node), grad_input(grad_input) {}

template <typename T>
void NodeTask<T>::update_grad_input(const Tensor<T> grad_input) {
    this->grad_input = this->grad_input + grad_input;
}
