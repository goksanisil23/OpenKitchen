#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Critic : torch::nn::Module
{
    static constexpr int32_t kStateDim{5}; // Normalized ray distances
    static constexpr int32_t kHiddenLayerSize1{128};
    static constexpr int32_t kOutputDim{1}; // Estimated value of being in this state

    // Critic network estimating the State->Value
    Critic()
    {
        l1 = register_module("l1", torch::nn::Linear(kStateDim, kHiddenLayerSize1));
        l2 = register_module("l2", torch::nn::Linear(kHiddenLayerSize1, kOutputDim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = l1->forward(x);
        x = torch::relu(x);
        x = l2->forward(x);
        return x;
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr};
};
