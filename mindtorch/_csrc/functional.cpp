#include "functional.h"
#include "grad_node.h"
#include "edge.h"
// 实现 add 函数


template <>
inline Tensor<int> add(Tensor<int>& t1, Tensor<int>& t2) {
    // 计算数据和 requires_grad
    auto out = t1 + t2;
    bool requires_grad = t1.getRequiresGrad() || t2.getRequiresGrad();
    std::cout << "Intermediate Value: " << requires_grad << std::endl;
    if (requires_grad) {
        // 创建 AddBackward 节点
        auto add_bw = new AddBackward<int>();
        add_bw->set_next_edges(collect_next_edges(t1, t2));

        // 设置 t1_shape 和 t2_shape
        if (t1.getRequiresGrad()) {
            add_bw->set_t1_shape(t1.getSize());
        }
        if (t2.getRequiresGrad()) {
            add_bw->set_t2_shape(t2.getSize());
        }

        // 返回带梯度信息的 Tensor
        return Tensor<int>(out.getData(), requires_grad, add_bw);
    } else {
        // 返回不带梯度信息的 Tensor
        return Tensor<int>(out.getData());
    }
}

// 实现辅助函数 collect_next_edges
template <typename T, typename... Args>
std::vector<Edge> collect_next_edges(Tensor<T>& t, Args&... tensors) {
    std::vector<Edge> next_edges;
    process_tensor(next_edges, t);
    process_tensor(next_edges, tensors...);
    return next_edges;
}

template <typename T>
void process_tensor(std::vector<Edge>& next_edges, Tensor<T>& t) {
    if (t.getRequiresGrad()) {
        auto grad_fn = t.getGradFn();
        if (grad_fn == nullptr) {
            std::cout << "grad_fn is null" << std::endl;
            auto grad_fn = new AccumulateGrad<T>(&t);
            t.setGradFn(grad_fn);
            next_edges.push_back(Edge(grad_fn));
        } else {
            next_edges.push_back(Edge(grad_fn));
        }
    }
}

template <typename T, typename... Args>
void process_tensor(std::vector<Edge>& next_edges, Tensor<T>& t, Args&... tensors) {
    process_tensor(next_edges, t);
    process_tensor(next_edges, tensors...);
}
