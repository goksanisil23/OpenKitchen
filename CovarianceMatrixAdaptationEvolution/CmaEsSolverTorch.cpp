#include "CmaEsSolverTorch.h"
#include <algorithm>
#include <cmath>
#include <iostream>

CmaEsSolver::CmaEsSolver(int num_params, int population_size)
    : CmaEsSolver(num_params, population_size, torch::Device(torch::kCPU))
{
}

CmaEsSolver::CmaEsSolver(int num_params, int population_size, torch::Device device)
    : num_params_(num_params), population_size_(population_size), num_parents_(population_size / 2), sigma_(0.5),
      device_(std::move(device))
{

    // init state on device_
    param_mean_ = torch::zeros({num_params_}, opts64());
    C_          = torch::eye(num_params_, opts64());
    p_sigma_    = torch::zeros({num_params_}, opts64());
    p_c_        = torch::zeros({num_params_}, opts64());

    // recombination weights (log scheme) on device_
    weights_ = torch::empty({num_parents_}, opts64());
    for (int i = 0; i < num_parents_; ++i)
        weights_[i] = std::log(num_parents_ + 0.5) - std::log(i + 1.0);
    weights_ /= weights_.sum();
    for (int i = 0; i < num_parents_; ++i)
        std::cout << "weight " << i << ": " << weights_[i].item<double>() << std::endl;

    mu_eff_ = 1.0 / weights_.pow(2).sum().item<double>();

    // learning rates
    c_sigma_ = (mu_eff_ + 2.0) / (num_params_ + mu_eff_ + 5.0);
    d_sigma_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff_ - 1.0) / (num_params_ + 1.0)) - 1.0) + c_sigma_;
    c_c_     = (4.0 + mu_eff_ / num_params_) / (num_params_ + 4.0 + 2.0 * mu_eff_ / num_params_);
    c_1_     = 2.0 / ((num_params_ + 1.3) * (num_params_ + 1.3) + mu_eff_);
    c_mu_    = std::min(1.0 - c_1_,
                     2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) / ((num_params_ + 2.0) * (num_params_ + 2.0) + mu_eff_));

    // E[||N(0,I)||]
    chiN_ = std::sqrt(static_cast<double>(num_params_)) *
            (1.0 - 1.0 / (4.0 * num_params_) + 1.0 / (21.0 * num_params_ * num_params_));

    // sampling cache
    B_ = torch::eye(num_params_, opts64());
    D_ = torch::ones({num_params_}, opts64());
}

std::vector<torch::Tensor> CmaEsSolver::sample()
{
    // SVD (C is symmetric PSD): C = U S V^T; eigvecs=U, eigvals=S
    auto svd = torch::svd(C_);
    B_       = std::get<0>(svd).contiguous();        // [N,N] on device_
    D_       = std::get<1>(svd).clamp_min(0).sqrt(); // [N]   on device_

    std::vector<torch::Tensor> out;
    out.reserve(population_size_);
    for (int i = 0; i < population_size_; ++i)
    {
        torch::Tensor z = torch::randn({num_params_}, opts64()); // N(0,I)
        torch::Tensor y = torch::matmul(B_, D_ * z);             // rotate/scale
        torch::Tensor x = param_mean_ + sigma_ * y;              // mean shift
        out.push_back(x.to(torch::kFloat32));                    // keep device_, cast to f32 for API parity
    }
    return out;
}

void CmaEsSolver::tell(const std::vector<SolutionAndFitness> &solutions)
{
    auto sorted = solutions;
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.fitness > b.fitness; });

    torch::Tensor old_mean = param_mean_.clone();
    param_mean_.zero_();

    for (int i = 0; i < num_parents_; ++i)
    {
        const double  wi = weights_[i].item<double>();
        torch::Tensor si = sorted[i].solution.to(device_, torch::kFloat64).contiguous();
        param_mean_.add_(si, wi);
    }

    torch::Tensor y_w = (param_mean_ - old_mean) / sigma_; // [N] on device_

    // p_sigma update using C^{-1/2} y_w ≈ B * D^{-1} * B^T * y_w
    torch::Tensor Bt_y   = torch::matmul(B_.t(), y_w);
    torch::Tensor scaled = Bt_y / D_.clamp_min(1e-12);
    torch::Tensor term   = torch::matmul(B_, scaled);

    p_sigma_ = p_sigma_ * (1.0 - c_sigma_) + term * std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_);

    // p_c update
    p_c_ = p_c_ * (1.0 - c_c_) + y_w * std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_);

    // rank-μ update
    torch::Tensor rank_mu = torch::zeros({num_params_, num_params_}, opts64());
    for (int i = 0; i < num_parents_; ++i)
    {
        const double  wi  = weights_[i].item<double>();
        torch::Tensor si  = sorted[i].solution.to(device_, torch::kFloat64).contiguous();
        torch::Tensor y_i = (si - old_mean) / sigma_;
        rank_mu.add_(y_i.view({-1, 1}).mm(y_i.view({1, -1})), wi);
    }

    C_ = (1.0 - c_1_ - c_mu_) * C_ + c_1_ * p_c_.view({-1, 1}).mm(p_c_.view({1, -1})) + c_mu_ * rank_mu;
    C_ = 0.5 * (C_ + C_.t()); // symmetrize

    const double ps_norm = p_sigma_.norm().item<double>();
    sigma_ *= std::exp((c_sigma_ / d_sigma_) * (ps_norm / chiN_ - 1.0));
}

torch::Tensor CmaEsSolver::get_best_solution() const
{
    return param_mean_.to(torch::kFloat32); // stays on device_
}
