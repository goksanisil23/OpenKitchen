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
    static constexpr int32_t kInputsSize{5}; // size of the states
    static constexpr int32_t kOutputSize{5}; // size of the action values
    static constexpr int32_t kHiddenLayerSize1{400};
    static constexpr int32_t kHiddenLayerSize2{300};

    Network()
    {
        l1 = register_module("l1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        l2 = register_module("l2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize2));
        l3 = register_module("l3", torch::nn::Linear(kHiddenLayerSize2, kOutputSize));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = l1->forward(x);
        x = torch::relu(x);
        x = l2->forward(x);
        x = torch::relu(x);
        x = l3->forward(x);

        // using raw q-value estimates, without activation. epsilon-greedy will handle choosing which action
        return x;
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
};
