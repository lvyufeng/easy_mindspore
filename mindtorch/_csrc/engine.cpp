// Engine.cpp

#include <map>
#include "engine.h"

template class Engine<int>; 

template <typename T>
void Engine<T>::execute(Tensor<T>& tensor, Tensor<T>& grad_input) {
    std::unordered_map<Node*, int> dependencies = compute_dependencies(tensor.getGradFn());
    std::map<int, NodeTask<T>> not_ready_dict;
    std::deque<NodeTask<T>> ready_queue = {{tensor.getGradFn(), grad_input}};

    while (!ready_queue.empty()) {
        NodeTask<T> node_task = ready_queue.at(0);
        ready_queue.pop_front();

        std::vector<Tensor<T>> grad_outputs = node_task.node->apply(node_task.grad_input);

        if (grad_outputs.empty()) {
            continue;
        }

        if (grad_outputs.size() != node_task.node->getNextEdges().size()){
        }
        for (size_t i = 0; i < grad_outputs.size(); ++i) {
            Node* next_node = node_task.node->getNextEdges()[i].getNode();
            dependencies[next_node] -= 1;


            auto iter = not_ready_dict.find(next_node->getId());
            if (iter == not_ready_dict.end()) {
                // 不存在键的情况，需要插入元素
                // not_ready_dict[next_node->getId()] = NodeTask<T>(next_node, grad_outputs[i]);
                not_ready_dict.insert({next_node->getId(), NodeTask<T>(next_node, grad_outputs[i])});

            } else {
                // 已存在键的情况，更新 grad_input
                iter->second.update_grad_input(grad_outputs[i]);
            }

            if (dependencies[next_node] == 0) {
                auto new_iter = not_ready_dict.find(next_node->getId());

                ready_queue.push_back(new_iter->second);
            }
        }
    }
}

template <typename T>
std::unordered_map<Node*, int> Engine<T>::compute_dependencies(Node* root) {
    std::unordered_map<Node*, int> dependencies;
    std::queue<Node*> queue;
    queue.push(root);

    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();

        if (node->hasNextEdges()) {
            for (const auto& edge : node->getNextEdges()) {
                Node* next_node = edge.getNode();
                dependencies[next_node] += 1;
                queue.push(next_node);
            }
        }
    }

    return dependencies;
}
