#include <iostream>
#include <torch/torch.h>

// Cost Network: Maps state [7] and action [2] to scalar cost
struct CostNet : torch::nn::Module
{
    CostNet()
    {
        fc1 = register_module("fc1", torch::nn::Linear(9, 64)); // 7 + 2
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
        fc3->weight.data().mul_(0.1);
        fc3->bias.data().mul_(0.0);
    }

    torch::Tensor forward(torch::Tensor s, torch::Tensor a)
    {
        torch::Tensor x = torch::cat({s, a}, /*dim=*/1);
        // x               = torch::relu(fc1->forward(x));
        // x               = torch::relu(fc2->forward(x));
        // x               = fc3->forward(x); // Scalar cost
        // return x;

        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        return fc3->forward(x);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
// Policy Network: Maps state [7] to action [2]
// with learnable log‑std
struct PolicyNet : torch::nn::Module
{
    PolicyNet()
    {
        fc1     = register_module("fc1", torch::nn::Linear(7, 64));
        fc2     = register_module("fc2", torch::nn::Linear(64, 64));
        fc3     = register_module("fc3", torch::nn::Linear(64, 2));
        log_std = register_parameter("log_std", torch::zeros({2})); // one per action dim
    }

    // returns (μ, logσ)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x       = torch::relu(fc1->forward(x));
        x       = torch::relu(fc2->forward(x));
        auto mu = torch::tanh(fc3->forward(x));
        return {mu, log_std};
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::Tensor     log_std;
};

struct ValueNet : torch::nn::Module
{
    ValueNet()
    {
        fc1 = register_module("fc1", torch::nn::Linear(7, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x).squeeze(1); // [B]
    }
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
