#include <iostream>
#include <vector>
#include "engine.cpp"

class Tensor;

// Example Node implementation
class ExampleNode : public Node {
public:
    ExampleNode() {}

    std::vector<double> apply(const std::vector<double>& grad_outputs) override {
        // Example implementation, just return the input for demonstration purposes
        return grad_outputs;
    }
};

// Example Tensor class
class Tensor {
public:
    Node* grad_fn;

    Tensor(Node* fn) : grad_fn(fn) {}
};


int main() {
    // Create nodes
    ExampleNode node1, node2, node3;
    
    // Connect nodes with edges
    node1.setNextEdges({&node2});
    node2.setNextEdges({&node3});

    // Create a tensor and specify the initial gradient
    Tensor tensor(&node1);
    std::vector<double> initial_grad = {1.0, 2.0, 3.0};
    
    // Create the Engine and execute the computational graph
    Engine engine;
    engine.execute(tensor.grad_fn, initial_grad);

    std::cout << "Test case executed successfully!" << std::endl;

    return 0;
}
