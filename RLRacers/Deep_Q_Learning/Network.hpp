// Neural Network used for approximating the value function in this reinforcement learning problem
// where the inputs are the raycast measurements and speed/rotation states of the robot
// and the outputs are the Q-values associated to each possible action

#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Network : torch::nn::Module
{
    // static constexpr int32_t kInputsFromSensor{15};
    // static constexpr int32_t kInputsSize{kInputsFromSensor + 2};
    static constexpr int32_t kInputsSize{5};
    static constexpr int32_t kOutputSize{3};
    static constexpr int32_t kHiddenLayerSize1{128};
    static constexpr int32_t kHiddenLayerSize2{128};

    Network()
    {
        // Fully connected layers
        affine1 = register_module("affine1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        affine2 = register_module("affine2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize2));
        affine3 = register_module("affine3", torch::nn::Linear(kHiddenLayerSize2, kOutputSize));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(affine1->forward(x));
        x = torch::relu(affine2->forward(x));
        x = affine3->forward(x);
        // using raw q-value estimates, without activation. epsilon-greedy will handle choosing which action
        return x;
    }

    torch::nn::Linear affine1{nullptr}, affine2{nullptr}, affine3{nullptr};
};
