#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include "node_task.h"
#include "node.h"

template <typename T>
class Engine {
public:
    void execute(Tensor<T>& tensor, Tensor<T>& grad_input);

private:
    std::unordered_map<Node*, int> compute_dependencies(Node* root);
};