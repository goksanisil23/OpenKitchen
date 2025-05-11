#pragma once

#include <cassert>
#include <iostream>
#include <torch/torch.h>

struct Policy : torch::nn::Module
{
    // static constexpr int32_t kInputsFromSensor{15};
    // static constexpr int32_t kInputsSize{kInputsFromSensor + 2};
    static constexpr int32_t kInputsSize{5}; // Normalized ray distances
    static constexpr int     kActionDim = 2; // throttle & steering
    static constexpr int32_t kHiddenLayerSize1{128};

    static constexpr bool kEnableLogStdHead{false}; // otherwise use log_std as a parameter

    Policy()
    {
        fc1  = register_module("fc1", torch::nn::Linear(kInputsSize, kHiddenLayerSize1));
        fc2  = register_module("fc2", torch::nn::Linear(kHiddenLayerSize1, kHiddenLayerSize1));
        mean = register_module("mean", torch::nn::Linear(kHiddenLayerSize1, kActionDim));
        if constexpr (kEnableLogStdHead)
        {
            log_std_head = register_module("log_std_head", torch::nn::Linear(kHiddenLayerSize1, kActionDim));
            register_parameter("log_std", torch::zeros({kActionDim}), /*requires_grad=*/false);
        }
        else
        {
            log_std = register_parameter("log_std", 2.5F * torch::ones({kActionDim}));
        }

        // log_std = register_parameter("log_std", 2.5F * torch::ones({kActionDim}));
        // log_std_head = register_module("log_std_head", torch::nn::Linear(kHiddenLayerSize1, kActionDim));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x                = torch::relu(fc1->forward(x));
        x                = torch::relu(fc2->forward(x));
        torch::Tensor mu = mean->forward(x);

        if constexpr (kEnableLogStdHead)
        {
            auto raw_ls          = log_std_head->forward(x);
            auto log_std_clamped = torch::clamp(raw_ls, /*min=*/-20, /*max=*/2);
            return {mu, log_std_clamped};
        }
        else
        {
            torch::Tensor log_std_expanded = log_std.expand_as(mu);
            return {mu, log_std_expanded};
        }
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean{nullptr};

    torch::nn::Linear log_std_head{nullptr};

    torch::Tensor log_std;

    // Containers for saved log probabilities and rewards
    std::vector<torch::Tensor> saved_log_probs_;
    std::vector<float>         rewards_;
};
