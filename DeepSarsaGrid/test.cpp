#include <iostream>
#include <torch/torch.h>

struct State
{
    float x, y;
};

using Action = int;

struct Net : torch::nn::Module
{
    static constexpr int64_t kHiddenLayerSize{60};
    Net()
        : fc1(register_module("fc1", torch::nn::Linear(2, kHiddenLayerSize))),
          fc2(register_module("fc2", torch::nn::Linear(kHiddenLayerSize, kHiddenLayerSize))),
          out(register_module("out", torch::nn::Linear(kHiddenLayerSize, 4)))
    {
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = out->forward(x);
        return x;
    }

    torch::nn::Linear fc1, fc2, out;
};

bool areTensorsEqual(const torch::Tensor &a, const torch::Tensor &b)
{
    return torch::allclose(a, b, 1e-9);
}

void test_grad_contrib()
{
    State  state       = {1.0, 2.0};
    State  next_state  = {2.0, 3.0};
    Action action      = 0;
    Action next_action = 1;
    double reward      = 5.0;
    bool   done        = false;

    // Setup a sample QNetwork and optimizer
    auto              q_network = std::make_shared<Net>();
    torch::optim::SGD optimizer(q_network->parameters(), /*lr=*/0.01);

    // Compute the gradient with next_q_values detached
    optimizer.zero_grad();
    torch::Tensor old_state_tensor       = torch::tensor({state.x, state.y}, torch::kFloat32);
    torch::Tensor new_state_tensor       = torch::tensor({next_state.x, next_state.y}, torch::kFloat32);
    auto          old_q_values           = q_network->forward(old_state_tensor);
    auto          next_q_values_detached = q_network->forward(new_state_tensor).detach();
    auto          target_detached        = old_q_values.clone().detach();
    target_detached[action]              = reward + 0.9 * next_q_values_detached[next_action].item<float>();
    torch::mse_loss(old_q_values, target_detached).backward();
    auto grad_detached = q_network->fc2->weight.grad().clone();

    // Compute the gradient as in the original function
    optimizer.zero_grad();
    old_q_values       = q_network->forward(old_state_tensor);
    auto next_q_values = q_network->forward(new_state_tensor);
    q_network->forward(new_state_tensor);
    q_network->forward(new_state_tensor * 2.F);
    q_network->forward(new_state_tensor * 3.F);
    q_network->forward(new_state_tensor * 4.F);
    // auto target    = old_q_values.clone().detach();
    auto target    = old_q_values.clone();
    target[action] = reward + 0.9 * next_q_values[next_action].item<float>();
    torch::mse_loss(old_q_values, target).backward();
    auto grad_normal = q_network->fc2->weight.grad();

    // Compare the gradients
    if (areTensorsEqual(grad_detached, grad_normal))
    {
        std::cout << "EQUAL." << std::endl;
    }
    else
    {
        std::cout << "NOT EQUAL." << std::endl;
    }
}

int main()
{
    test_grad_contrib();
    return 0;
}
