#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Net : torch::nn::Module
{
    Net(int input_size, int hidden_size, int output_size)
    {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

void test_Net()
{
    // 1. Create a neural network instance
    int input_size  = 8;
    int hidden_size = 16;
    int output_size = 4;
    Net net(input_size, hidden_size, output_size);

    // 2. Test forward pass
    torch::Tensor sample_input = torch::randn({1, input_size});
    torch::Tensor output       = net.forward(sample_input);

    // Assert that output size is correct
    assert(output.size(0) == 1);
    assert(output.size(1) == output_size);

    // 3. Test training on dummy data
    auto          optimizer    = torch::optim::Adam(net.parameters(), 0.001);
    torch::Tensor dummy_target = torch::randn({1, output_size});

    for (int epoch = 0; epoch < 10; ++epoch)
    {
        optimizer.zero_grad();
        torch::Tensor prediction = net.forward(sample_input);
        torch::Tensor loss       = torch::mse_loss(prediction, dummy_target);
        loss.backward();
        optimizer.step();
    }

    // Check that weights are being updated
    torch::Tensor new_output = net.forward(sample_input);
    std::cout << output << std::endl;
    std::cout << new_output << std::endl;
    assert(!torch::allclose(output, new_output));

    std::cout << "Net test passed!" << std::endl;
}

int main()
{
    test_Net();
    return 0;
}
