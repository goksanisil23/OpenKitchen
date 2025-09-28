#include "Controller.h"

Controller::Controller(int64_t input_size, int64_t hidden_size, int64_t output_size)
{
    fc1_ = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
    fc2_ = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
}

torch::Tensor Controller::forward(torch::Tensor x)
{
    x = torch::tanh(fc1_->forward(x));
    x = torch::tanh(fc2_->forward(x)); // Tanh activation to bound outputs to [-1, 1]
    return x;
}

int64_t Controller::count_params()
{
    int64_t num_params = 0;
    for (const auto &p : parameters())
    {
        num_params += p.numel();
    }
    return num_params;
}

torch::Tensor Controller::get_flat_params()
{
    std::vector<torch::Tensor> params;
    for (const auto &p : parameters())
    {
        params.push_back(p.clone().detach().flatten());
    }
    return torch::cat(params);
}

void Controller::set_params(const torch::Tensor &flat_params)
{
    torch::NoGradGuard no_grad;
    int64_t            offset = 0;
    for (auto &p : parameters())
    {
        int64_t numel = p.numel();
        p.copy_(flat_params.slice(0, offset, offset + numel).view_as(p));
        offset += numel;
    }
}