#include "CmaEsSolverEigen.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace
{

// Helper to convert Eigen::VectorXd to torch::Tensor
torch::Tensor eigen_to_tensor(const Eigen::VectorXd &vec)
{
    return torch::from_blob(const_cast<double *>(vec.data()), {vec.size()}, torch::kFloat64)
        .clone()
        .to(torch::kFloat32);
}

// Helper to convert torch::Tensor to Eigen::VectorXd
Eigen::VectorXd tensor_to_eigen(const torch::Tensor &tensor)
{
    torch::Tensor tensor_d = tensor.to(torch::kFloat64).contiguous();
    return Eigen::Map<Eigen::VectorXd>(tensor_d.data_ptr<double>(), tensor_d.numel());
}
} // namespace

CmaEsSolver::CmaEsSolver(const int num_params, const int population_size)
    : num_params_(num_params), population_size_(population_size), num_parents_(population_size / 2), sigma_(0.5)
{
    // Initialize mean at zero
    param_mean_ = Eigen::VectorXd::Zero(num_params_);

    // Initialize covariance and evolution paths
    C_       = Eigen::MatrixXd::Identity(num_params_, num_params_);
    p_sigma_ = Eigen::VectorXd::Zero(num_params_);
    p_c_     = Eigen::VectorXd::Zero(num_params_);

    // Set up recombination weights: In each generation, top scorers are weighted according to this predefined scheme
    weights_ = Eigen::VectorXd(num_parents_);
    for (int i = 0; i < num_parents_; ++i)
    {
        weights_(i) = std::log(num_parents_ + 0.5) - std::log(i + 1);
    }
    weights_ /= weights_.sum();
    for (auto i = 0; i < weights_.size(); i++)
    {
        std::cout << "weight " << i << ": " << weights_(i) << std::endl;
    }
    mu_eff_ = 1.0 / weights_.squaredNorm();

    // Set up learning rates
    c_sigma_ = (mu_eff_ + 2.0) / (num_params_ + mu_eff_ + 5.0);
    d_sigma_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff_ - 1.0) / (num_params_ + 1.0)) - 1.0) + c_sigma_;
    c_c_     = (4.0 + mu_eff_ / num_params_) / (num_params_ + 4.0 + 2.0 * mu_eff_ / num_params_);
    c_1_     = 2.0 / ((num_params_ + 1.3) * (num_params_ + 1.3) + mu_eff_);
    c_mu_    = std::min(1.0 - c_1_,
                     2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) / ((num_params_ + 2.0) * (num_params_ + 2.0) + mu_eff_));

    // Pre-calculate E[||N(0,I)||]
    chiN_ = std::sqrt(num_params_) * (1.0 - 1.0 / (4.0 * num_params_) + 1.0 / (21.0 * num_params_ * num_params_));

    // Initialize B and D for sampling
    B_ = Eigen::MatrixXd::Identity(num_params_, num_params_);
    D_ = Eigen::VectorXd::Ones(num_params_);
}

std::vector<torch::Tensor> CmaEsSolver::sample()
{
    // Eigendecomposition of C for sampling: C = B * D * D * B^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(C_);
    B_ = eigensolver.eigenvectors();            // rotation
    D_ = eigensolver.eigenvalues().cwiseSqrt(); // scaling

    std::vector<torch::Tensor> solutions;
    for (int i = 0; i < population_size_; ++i)
    {
        // Sample from N(0, I)
        Eigen::VectorXd z = Eigen::VectorXd::NullaryExpr(num_params_, [&]() { return norm_dist_(rand_generator_); });

        // Transform z to sample from N(mean, sigma^2 * C)
        Eigen::VectorXd y = B_ * (D_.asDiagonal() * z);
        Eigen::VectorXd x = param_mean_ + sigma_ * y;

        solutions.push_back(eigen_to_tensor(x));
    }
    return solutions;
}

// https://arxiv.org/pdf/1604.00772
void CmaEsSolver::tell(const std::vector<SolutionAndFitness> &solutions)
{
    // Sort solutions by fitness (descending order)
    auto sorted_solutions = solutions;
    std::sort(sorted_solutions.begin(),
              sorted_solutions.end(),
              [](const auto &a, const auto &b) { return a.fitness > b.fitness; });

    // --- Update weighted mean (over parents) ---
    const Eigen::VectorXd old_mean = param_mean_;
    param_mean_.setZero();
    for (int i = 0; i < num_parents_; ++i)
    {
        param_mean_ += weights_(i) * tensor_to_eigen(sorted_solutions[i].solution);
    }

    const Eigen::VectorXd y_w = (param_mean_ - old_mean) / sigma_;

    // --- Update Evolution Paths ---
    // [Equation 31]
    p_sigma_ = (1.0 - c_sigma_) * p_sigma_ +
               std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_) * B_ * D_.asDiagonal().inverse() * B_.transpose() * y_w;

    // [Equation 24]
    p_c_ = (1.0 - c_c_) * p_c_ + std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_) * y_w;

    // --- Update Covariance Matrix C ---
    // [Equation 30]
    Eigen::MatrixXd rank_mu_update = Eigen::MatrixXd::Zero(num_params_, num_params_);
    for (int i = 0; i < num_parents_; ++i)
    {
        Eigen::VectorXd y_i = (tensor_to_eigen(sorted_solutions[i].solution) - old_mean) / sigma_;
        rank_mu_update += weights_(i) * y_i * y_i.transpose();
    }

    C_ = (1.0 - c_1_ - c_mu_) * C_ + c_1_ * (p_c_ * p_c_.transpose()) + c_mu_ * rank_mu_update;

    // --- Update Step-Size sigma ---
    // [Equation 37]
    sigma_ *= std::exp((c_sigma_ / d_sigma_) * (p_sigma_.norm() / chiN_ - 1.0));
}

torch::Tensor CmaEsSolver::get_best_solution() const
{
    return eigen_to_tensor(param_mean_);
}