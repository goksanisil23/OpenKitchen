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

    static constexpr float kOut1Max{50}; // direct m/s setting
    static constexpr float kOut2Max{5};  // steering delta degrees

    // Actor network estimating the State->Action
    Actor()
    {
        l1 = register_module("l1", torch::nn::Linear(kStateDim, kHiddenLayerSize1));
        l2 = register_module("l2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize2));
        l3 = register_module("l3", torch::nn::Linear(kHiddenLayerSize2, kActionDim));

        scale = register_buffer("scale", torch::tensor({kOut1Max, kOut2Max}, torch::dtype(torch::kFloat32)));
        bias  = register_buffer("bias", torch::tensor({kOut1Max, 0.0f}, torch::dtype(torch::kFloat32)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = l1->forward(x);
        x = torch::relu(x);
        x = l2->forward(x);
        x = torch::relu(x);
        x = l3->forward(x);
        x = torch::tanh(x);      // [-1,1]
        return x * scale + bias; // to map from [-50,50] to [0,100]
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    torch::Tensor     scale, bias;
};
