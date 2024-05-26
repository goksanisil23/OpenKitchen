#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Actor : torch::nn::Module
{
    static constexpr int32_t kInputsSize{5}; // Normalized ray distances
    static constexpr int32_t kOutputSize{3}; // Probability per each action
    static constexpr int32_t kHiddenLayerSize1{128};

    // Actor network estimating the State->Action-Probs
    Actor()
    {
        l1 = register_module("l1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        l2 = register_module("l2", torch::nn::Linear(kHiddenLayerSize1, kOutputSize));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = l1->forward(x);
        x = torch::relu(x);
        x = l2->forward(x);
        return torch::softmax(x, /*dim=*/1);
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr};
};
