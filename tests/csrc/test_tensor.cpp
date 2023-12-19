// test/main.cpp
#include "../../mindtorch/_csrc/tensor.h"  // 包含正确的路径

int main() {
    // 测试代码
    Tensor<int> scalar1(5);
    Tensor<int> scalar2(10);
    Tensor<int> result_scalar_add = scalar1 + scalar2;
    Tensor<int> result_scalar_mul = scalar1 * scalar2;

    std::cout << "Scalar Addition: " << result_scalar_add.getData() << std::endl;
    std::cout << "Scalar Multiplication: " << result_scalar_mul.getData() << std::endl;

    std::vector<int> list1 = {1, 2, 3};
    std::vector<int> list2 = {4, 5, 6};
    Tensor<std::vector<int>> tensor_list1(list1);
    Tensor<std::vector<int>> tensor_list2(list2);

    Tensor<std::vector<int>> result_list_add = tensor_list1 + tensor_list2;
    Tensor<std::vector<int>> result_list_mul = tensor_list1 * tensor_list2;

    std::cout << "List Addition: ";
    for (const auto& element : result_list_add.getData()) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    std::cout << "List Multiplication: ";
    for (const auto& element : result_list_mul.getData()) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    return 0;
}
