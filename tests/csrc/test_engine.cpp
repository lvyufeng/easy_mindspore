// main.cpp

#include "../../mindtorch/_csrc/tensor.h"
#include "../../mindtorch/_csrc/node.h"
#include "../../mindtorch/_csrc/functional.h"
#include "../../mindtorch/_csrc/node_task.cpp"
#include "../../mindtorch/_csrc/engine.cpp"

int main() {
    // 创建两个 Tensor，并设置 requires_grad 为 true
    Tensor<int> t1(2, true);
    Tensor<int> t2(3, true);

    // 执行加法操作，得到结果 Tensor
    Tensor<int> result = add(t1, t2);

    // 定义初始梯度 grad_input 为 1
    Tensor<int> grad_input(1);

    // 创建 Engine 实例，执行梯度计算
    Engine<int> engine;
    engine.execute(result, grad_input);

    // 输出梯度值
    std::cout << "t1 grad value: " << (t1.getGrad() ? "not null": "null") << std::endl;
    std::cout << "t1 grad_fn: " << (t1.getGradFn() ? "not null": "null") << std::endl;

    std::cout << "t1 grad: " << t1.getGrad()->getData() << std::endl;
    std::cout << "t2 grad: " << t2.getGrad()->getData() << std::endl;

    return 0;
}
