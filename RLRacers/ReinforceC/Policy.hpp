#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Policy : torch::nn::Module
{
    // static constexpr int32_t kInputsFromSensor{15};
    // static constexpr int32_t kInputsSize{kInputsFromSensor + 2};
    static constexpr int32_t kInputsSize{5}; // Normalized ray distances
    static constexpr int     kActionDim = 2; // throttle & steering
    static constexpr int32_t kHiddenLayerSize1{128};

    Policy()
    {
        fc1  = register_module("fc1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        fc2  = register_module("fc2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize1));
        mean = register_module("mean", torch::nn::Linear(kHiddenLayerSize1, kActionDim));
        // log_std = register_parameter("log_std", torch::ones({kActionDim})) * 0.5F;
        log_std = register_parameter("log_std", torch::ones({kActionDim}) * 0.5F);
    }

    // returns pair<mean, log_std>, each [batch x ActionDim]
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x                              = torch::relu(fc1->forward(x));
        x                              = torch::relu(fc2->forward(x));
        torch::Tensor mu               = mean->forward(x);
        torch::Tensor log_std_expanded = log_std.expand_as(mu);
        return {mu, log_std_expanded};
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean{nullptr};
    torch::Tensor     log_std;

    // Containers for saved log probabilities and rewards
    std::vector<torch::Tensor> saved_log_probs;
    std::vector<float>         rewards;
};
