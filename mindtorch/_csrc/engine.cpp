#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include "tensor.h"

class Edge;

// Abstract base class for Node
class Node {
public:
    virtual std::vector<Tensor> apply(Tensor& grad_outputs) = 0;
    void setNextEdges(const std::vector<Edge>& next_edges) {
        nextEdges = next_edges;
    }
    const std::vector<Edge>& getNextEdges() const {
        return nextEdges;
    }

protected:
    std::vector<Edge> nextEdges;
};

// Edge class
class Edge {
public:
    Node* node;
    Edge() : node(nullptr) {}
    Edge(Node* n) : node(n) {}
};
// NodeTask class
class NodeTask {
public:
    Node* node;
    Tensor gradInput;

    // Parameterized constructor
    NodeTask(Node* n, const Tensor& grad_input) : node(n), gradInput(grad_input) {}

    void updateGradInput(const Tensor& grad_input) {
        for (size_t i = 0; i < this->gradInput.size(); ++i) {
            this->gradInput[i] += grad_input[i];
        }
    }
};

// Engine class
class Engine {
public:
    void execute(Node* tensor, const Tensor& grad_input) {
        std::unordered_map<Node*, int> dependencies = computeDependencies(tensor);
        std::unordered_map<Node*, NodeTask> notReadyDict;
        std::deque<NodeTask> readyQueue = {NodeTask(tensor, grad_input)};

        while (!readyQueue.empty()) {
            NodeTask nodeTask = readyQueue.front();
            readyQueue.pop_front();
            std::vector<Tensor> gradOutputs = nodeTask.node->apply(nodeTask.gradInput);

            if (gradOutputs.empty()) {
                continue;
            }

            for (size_t i = 0; i < gradOutputs.size(); ++i) {
                std::cout << i << std::endl;

                Edge edge = nodeTask.node->getNextEdges()[i];
                Node* nextNode = edge.node;

                if (!nextNode){
                    continue;
                }
                dependencies[nextNode] -= 1;

                if (notReadyDict.find(nextNode) == notReadyDict.end()) {
                    notReadyDict[nextNode] = NodeTask(nextNode, gradOutputs[i]);
                } else {
                    notReadyDict[nextNode].updateGradInput(gradOutputs[i]);
                }

                if (dependencies[nextNode] == 0) {
                    readyQueue.push_back(notReadyDict[nextNode]);
                }
            }
        }
    }

private:
    std::unordered_map<Node*, int> computeDependencies(Node* root) {
        std::unordered_map<Node*, int> dependencies;
        dependencies[root] = 0;
        std::deque<Node*> queue = {root};

        while (!queue.empty()) {
            Node* node = queue.back();
            queue.pop_back();

            for (Edge edge : node->getNextEdges()) {
                Node* nextNode = edge.node;
                dependencies[nextNode] += 1;
                queue.push_back(nextNode);
            }
        }

        return dependencies;
    }
};
