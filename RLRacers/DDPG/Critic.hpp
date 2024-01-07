#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Critic : torch::nn::Module
{
    static constexpr int32_t kStateDim{5};
    static constexpr int32_t kActionDim{2};
    static constexpr int32_t kHiddenLayerSize1{400};
    static constexpr int32_t kHiddenLayerSize2{300};

    static constexpr float kOut1Max{100}; // direct m/s setting
    static constexpr float kOut2Max{10};  // steering delta degrees

    // Critic network estimating the {State,Action}->Value
    Critic()
    {
        l1 = register_module("l1", torch::nn::Linear(kStateDim + kActionDim, kHiddenLayerSize1));
        l2 = register_module("l2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize2));
        l3 = register_module("l3", torch::nn::Linear(kHiddenLayerSize2, 1));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action)
    {
        auto x = torch::cat({state, action}, 1);
        x      = l1->forward(x);
        x      = torch::relu(x);
        x      = l2->forward(x);
        x      = torch::relu(x);
        x      = l3->forward(x);
        return x;
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
};
