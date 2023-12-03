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
    static constexpr int32_t kHiddenLayerSize1{30};
    static constexpr int32_t kHiddenLayerSize2{30};

    Network()
    {
        // Fully connected layers
        fc1 = register_module("fc1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        fc2 = register_module("fc2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize2));
        fc3 = register_module("fc3", torch::nn::Linear(kHiddenLayerSize2, kOutputSize));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // activation function
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        // using raw q-value estimates, without activation. epsilon-greedy will handle choosing which action
        x = fc3->forward(x);
        return x;
    }

    void print_weights()
    {
        // for (const auto &pair : this->named_parameters())
        // {
        //     std::cout << pair.key() << ": " << pair.value() << std::endl;
        // }
        auto fc1_weights = this->fc1->weight;
        auto fc1_bias    = this->fc1->bias;

        std::cout << "FC1 Weights: " << fc1_weights << std::endl;
        std::cout << "FC1 Bias: " << fc1_bias << std::endl;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
