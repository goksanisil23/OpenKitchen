#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Policy : torch::nn::Module
{
    // static constexpr int32_t kInputsFromSensor{15};
    // static constexpr int32_t kInputsSize{kInputsFromSensor + 2};
    static constexpr int32_t kInputsSize{5};
    static constexpr int32_t kOutputSize{3};
    static constexpr int32_t kHiddenLayerSize1{128};

    Policy()
    {
        affine1 = register_module("affine1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        dropout = register_module("dropout", torch::nn::Dropout(0.6));
        affine2 = register_module("affine2", torch::nn::Linear(kHiddenLayerSize1, kOutputSize));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = affine1->forward(x);
        x = dropout->forward(x);
        x = torch::relu(x);
        x = affine2->forward(x);
        return torch::softmax(x, /*dim=*/1);
    }

    torch::nn::Linear  affine1{nullptr}, affine2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    // Containers for saved log probabilities and rewards
    std::vector<torch::Tensor> saved_log_probs;
    std::vector<float>         rewards;
};
