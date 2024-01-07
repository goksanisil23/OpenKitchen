#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Actor : torch::nn::Module
{
    // static constexpr int32_t kInputsFromSensor{15};
    // static constexpr int32_t kInputsSize{kInputsFromSensor + 2};
    static constexpr int32_t kStateDim{5};
    static constexpr int32_t kActionDim{2};
    static constexpr int32_t kHiddenLayerSize1{400};
    static constexpr int32_t kHiddenLayerSize2{300};

    static constexpr float kOut1Max{500}; // direct m/s setting
    static constexpr float kOut2Max{10};  // steering delta degrees

    // Actor network estimating the State->Action
    Actor()
    {
        l1 = register_module("l1", torch::nn::Linear(kStateDim, kHiddenLayerSize1));
        l2 = register_module("l2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize2));
        l3 = register_module("l3", torch::nn::Linear(kHiddenLayerSize2, kActionDim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = l1->forward(x);
        x = torch::relu(x);
        x = l2->forward(x);
        x = torch::relu(x);
        x = l3->forward(x);
        x = torch::tanh(x); // [-1,1]
        return x * torch::tensor({kOut1Max, kOut2Max});
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
};
