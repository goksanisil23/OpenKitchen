#include <torch/torch.h>

struct RewardNet : torch::nn::Module
{
    torch::nn::Sequential net;

    RewardNet()
    {
        // 9 = 7 (state) + 2 (action)
        net = torch::nn::Sequential(torch::nn::Linear(9, 64),
                                    torch::nn::ReLU(),
                                    torch::nn::Linear(64, 64),
                                    torch::nn::ReLU(),
                                    torch::nn::Linear(64, 1));
        register_module("net", net);
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action)
    {
        auto input = torch::cat({state, action}, /*dim=*/1);
        return net->forward(input);
    }
};

struct PolicyNet : torch::nn::Module
{
    // static constexpr float kOut1Max{50}; // direct m/s setting
    // static constexpr float kOut2Max{10}; // steering delta degrees

    torch::nn::Sequential net;

    PolicyNet()
    {
        net = torch::nn::Sequential(torch::nn::Linear(7, 64),
                                    torch::nn::ReLU(),
                                    torch::nn::Linear(64, 64),
                                    torch::nn::ReLU(),
                                    torch::nn::Linear(64, 2),
                                    torch::nn::Tanh() // assuming action range [-1, 1]
        );
        register_module("net", net);
    }

    torch::Tensor forward(torch::Tensor state)
    {
        return net->forward(state);
    }
};
