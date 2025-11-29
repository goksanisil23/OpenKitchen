#pragma once

#include <torch/torch.h>
#include <vector>

// A simple feed-forward neural network
class Controller : public torch::nn::Module
{
  public:
    Controller(const int64_t input_size, const int64_t hidden_size, const int64_t output_size);

    // The forward pass
    torch::Tensor forward(torch::Tensor x);

    // Helper function to count the total number of parameters (weights + biases)
    int64_t count_params();

    // Helper to get all parameters as a single flat tensor
    torch::Tensor get_flat_params();

    // Helper to set the network's parameters from a flat tensor
    void set_params(const torch::Tensor &flat_params);

  private:
    torch::nn::Linear fc1_{nullptr};
    // torch::nn::Linear fc2_{nullptr};
    // torch::nn::Linear fc3_{nullptr};
};
