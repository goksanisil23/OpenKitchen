#pragma once

#include <random>
#include <torch/torch.h>
#include <vector>

struct SolutionAndFitness
{
    torch::Tensor solution;
    float         fitness;
};

class CmaEsSolver
{
  public:
    CmaEsSolver(int num_params, int population_size, torch::Device device = torch::kCPU, const float sigma = 0.5F);

    std::vector<torch::Tensor> sample();
    void                       tell(const std::vector<SolutionAndFitness> &solutions);
    torch::Tensor              get_best_solution() const;

  private:
    // sizes
    const int num_params_;
    const int population_size_;
    const int num_parents_;

    // step-size
    float sigma_;

    // state (float64 on device_)
    torch::Tensor param_mean_; // [N]
    torch::Tensor C_;          // [N,N]
    torch::Tensor p_sigma_;    // [N]
    torch::Tensor p_c_;        // [N]
    torch::Tensor weights_;    // [mu]

    // constants
    float mu_eff_;
    float c_sigma_, d_sigma_;
    float c_c_, c_1_, c_mu_;
    float chiN_;

    // sampling cache
    torch::Tensor B_; // eigenvectors of C, [N,N]
    torch::Tensor D_; // eigenvalues of C, [N]

    // device
    torch::Device device_{torch::kCPU};

    inline torch::TensorOptions opts32() const
    {
        return torch::dtype(torch::kFloat32).device(device_);
    }

    // kept for compatibility
    std::mt19937                    rand_generator_{std::random_device{}()};
    std::normal_distribution<float> norm_dist_{0.0, 1.0};
};
