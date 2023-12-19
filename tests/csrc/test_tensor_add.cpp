#include <iostream>
#include "../../mindtorch/_csrc/tensor.h"
#include "../../mindtorch/_csrc/grad_node.h"
#include "../../mindtorch/_csrc/functional.h"

int main() {
    // 创建两个张量，设置 requires_grad 为 true
    Tensor<int> t1(3, true);
    Tensor<int> t2(5, true);

    Tensor<int> result1 = t1 + t2;
    std::cout << "t1 shape: " << t1.getSize() << std::endl;
    std::cout << "t2 shape: " << t2.getSize() << std::endl;

    std::cout << "Result: " << result1.getData() << std::endl;
    // 调用 add 函数进行相加
    Tensor<int> result = add(t1, t2);

    // 检查结果值
    std::cout << "Result: " << result.getData() << std::endl;

    // 检查是否设置了 requires_grad
    std::cout << "Requires Grad: " << std::boolalpha << result.getRequiresGrad() << std::endl;

    std::cout << "Dtype: " << result.getDataType() << std::endl;

    // 获取 grad_node，检查是否为 nullptr
    auto grad_fn = result.getGradFn();
    if (grad_fn == nullptr) {
        std::cerr << "Error: grad_fn is nullptr." << std::endl;
    }
    std::cout << "Grad Node: " << (grad_fn ? "Not nullptr" : "nullptr") << std::endl;


    auto backward_out = grad_fn->apply(t1);
    std::cout << "Result: " << backward_out.empty() << std::endl;

    return 0;
}