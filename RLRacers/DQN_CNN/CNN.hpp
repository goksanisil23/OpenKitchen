// Neural Network used for approximating the value function in this reinforcement learning problem
// where the inputs are the birdseye view images from the robot
// and the outputs are the Q-values associated to each possible action

#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct CNN : torch::nn::Module
{
    static constexpr size_t kImageWidth{1600};
    static constexpr size_t kImageHeight{1400};
    static constexpr size_t kImageChannels{4};

    static constexpr int32_t kOutputSize{5}; // size of the action values
    // static constexpr int32_t kHiddenLayerSize1{400};
    // static constexpr int32_t kHiddenLayerSize2{300};

    CNN()
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 8).stride(4).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
        fc1   = register_module("fc1", torch::nn::Linear(64 * 43 * 50, 512)); // Adjusted size
        fc2   = register_module("fc2", torch::nn::Linear(512, 5));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(conv1->forward(x));
        // std::cout << "After conv1: " << x.sizes() << std::endl;
        x = torch::relu(conv2->forward(x));
        // std::cout << "After conv2: " << x.sizes() << std::endl;
        x = torch::relu(conv3->forward(x));
        // std::cout << "After conv3: " << x.sizes() << std::endl;
        x = x.view({x.size(0), -1}); // Flatten
        // std::cout << "After flatten: " << x.sizes() << std::endl;
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
