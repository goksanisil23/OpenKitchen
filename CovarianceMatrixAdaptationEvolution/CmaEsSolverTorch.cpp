#include "CmaEsSolverTorch.h"
#include <algorithm>
#include <cmath>
#include <iostream>

CmaEsSolver::CmaEsSolver(int num_params, int population_size, const torch::Device device, const float sigma)
    : num_params_(num_params), population_size_(population_size), num_parents_(population_size / 2), sigma_(sigma),
      device_(device)
{

    // Initialize mean at zero
    param_mean_ = torch::zeros({num_params_}, opts32());

    // Initialize covariance and evolution paths
    C_       = torch::eye(num_params_, opts32());
    p_sigma_ = torch::zeros({num_params_}, opts32());
    p_c_     = torch::zeros({num_params_}, opts32());

    // Set up recombination weights: In each generation, top scorers are weighted according to this predefined scheme
    weights_ = torch::empty({num_parents_}, opts32());
    for (int i = 0; i < num_parents_; ++i)
    {
        weights_[i] = std::log(num_parents_ + 0.5) - std::log(i + 1.0);
    }
    weights_ /= weights_.sum();
    for (int i = 0; i < num_parents_; ++i)
    {
        std::cout << "weight " << i << ": " << weights_[i].item<float>() << std::endl;
    }
    mu_eff_ = 1.0 / weights_.pow(2).sum().item<float>();

    // learning rates
    c_sigma_ = (mu_eff_ + 2.0) / (num_params_ + mu_eff_ + 5.0);
    d_sigma_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff_ - 1.0) / (num_params_ + 1.0)) - 1.0) + c_sigma_;
    c_c_     = (4.0 + mu_eff_ / num_params_) / (num_params_ + 4.0 + 2.0 * mu_eff_ / num_params_);
    c_1_     = 2.0 / ((num_params_ + 1.3) * (num_params_ + 1.3) + mu_eff_);
    c_mu_    = std::min(1.0 - c_1_,
                     2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) / ((num_params_ + 2.0) * (num_params_ + 2.0) + mu_eff_));

    // Pre-calculate E[||N(0,I)||]
    chiN_ = std::sqrt(num_params_) * (1.0 - 1.0 / (4.0 * num_params_) + 1.0 / (21.0 * num_params_ * num_params_));

    // Initialize B and D for sampling
    B_ = torch::eye(num_params_, opts32());
    D_ = torch::ones({num_params_}, opts32());
}

std::vector<torch::Tensor> CmaEsSolver::sample()
{
    // SVD (C is symmetric PSD): C = U S V^T; eigvecs=U, eigvals=S
    //  C = B * D * D * B^T
    // std::cout << "starting svd" << std::endl;
    // auto svd = torch::svd(C_);
    // B_       = std::get<0>(svd).contiguous();                        // rotation
    // D_       = std::get<1>(svd).clamp_min(1e-8).sqrt().contiguous(); // scaling
    // std::cout << "finished svd" << std::endl;

    // auto eig_cpu = torch::linalg_eigh(C_.to(torch::kCPU));
    // C_ is supposed to be symmetric, but numerical errors might make it slightly non-symmetric
    C_           = (C_ + C_.t()) / 2.0;
    auto eig_cpu = torch::linalg_eigh(C_);
    // auto evals   = std::get<0>(eig_cpu).clamp_min(1e-12).sqrt().to(device_);
    // auto evecs   = std::get<1>(eig_cpu).to(device_).contiguous();
    auto evals = std::get<0>(eig_cpu).clamp_min(1e-12).sqrt();
    auto evecs = std::get<1>(eig_cpu).contiguous();

    // Assign
    D_ = evals;
    B_ = evecs;

    std::vector<torch::Tensor> solutions;
    solutions.reserve(population_size_);
    for (int i = 0; i < population_size_; ++i)
    {
        torch::Tensor z = torch::randn({num_params_}, opts32()); // N(0,I)
        torch::Tensor y = torch::matmul(B_, D_ * z);             // rotate/scale
        torch::Tensor x = param_mean_ + sigma_ * y;              // mean shift
        solutions.push_back(x.to(torch::kFloat32));
    }
    return solutions;
}

// https://arxiv.org/pdf/1604.00772
void CmaEsSolver::tell(const std::vector<SolutionAndFitness> &solutions)
{
    auto sorted = solutions;
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.fitness > b.fitness; });

    // --- Update weighted mean (over parents) ---
    torch::Tensor old_mean = param_mean_.clone();
    param_mean_.zero_();

    for (int i = 0; i < num_parents_; ++i)
    {
        const float         weight   = weights_[i].item<float>();
        torch::Tensor const solution = sorted[i].solution.to(device_, torch::kFloat64).contiguous();
        param_mean_ += weight * solution;
    }

    torch::Tensor y_w = (param_mean_ - old_mean) / sigma_;

    // --- Update Evolution Paths ---
    // [Equation 31]
    p_sigma_ = (1.0 - c_sigma_) * p_sigma_ +
               std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_) * torch::matmul(B_, (torch::matmul(B_.t(), y_w)) / D_);

    // [Equation 24]
    p_c_ = (1.0 - c_c_) * p_c_ + std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_) * y_w;

    // --- Update Covariance Matrix C ---
    // [Equation 30]
    torch::Tensor rank_mu_update = torch::zeros({num_params_, num_params_}, opts32());
    for (int i = 0; i < num_parents_; ++i)
    {
        const float         weight   = weights_[i].item<float>();
        torch::Tensor const solution = sorted[i].solution.to(device_, torch::kFloat64).contiguous();
        torch::Tensor       y_i      = (solution - old_mean) / sigma_;
        rank_mu_update += weight * (y_i.view({-1, 1}) * y_i.view({1, -1}));
    }

    // Should be symmetric by construction
    C_ = (1.0 - c_1_ - c_mu_) * C_ + c_1_ * (p_c_.view({-1, 1}) * p_c_.view({1, -1})) + c_mu_ * rank_mu_update;

    // --- Update Step-Size sigma ---
    // [Equation 37]
    const float ps_norm = p_sigma_.norm().item<float>();
    sigma_ *= std::exp((c_sigma_ / d_sigma_) * (ps_norm / chiN_ - 1.0));
}

torch::Tensor CmaEsSolver::get_best_solution() const
{
    return param_mean_.to(torch::kFloat32); // stays on device_
}
