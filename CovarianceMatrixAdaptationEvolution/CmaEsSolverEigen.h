#pragma once

#include <Eigen/Dense>
#include <random>
#include <torch/torch.h>
#include <vector>

struct SolutionAndFitness
{
    torch::Tensor solution;
    double        fitness;
};

class CmaEsSolver
{
  public:
    CmaEsSolver(const int num_params, const int population_size);

    // Ask for a new population of candidate solutions
    std::vector<torch::Tensor> sample();

    // Tell the solver the fitness of each candidate solution
    void tell(const std::vector<SolutionAndFitness> &solutions);

    // Get the current best estimate of the solution
    torch::Tensor get_best_solution() const;

  private:
    // CMA-ES state variables
    const int num_params_;      // number of parameters to optimize
    const int population_size_; // Population size
    const int num_parents_;     // Number of parents to select (number of best performers in each generation)

    double          sigma_;      // Step-size
    Eigen::VectorXd param_mean_; // Weighted mean of the parents' parameters
    Eigen::MatrixXd C_;          // Covariance matrix
    Eigen::VectorXd p_sigma_;    // Evolution path for sigma
    Eigen::VectorXd p_c_;        // Evolution path for C
    Eigen::VectorXd weights_;

    // CMA-ES algorithm constants
    double mu_eff_;
    double c_sigma_, d_sigma_;
    double c_c_, c_1_, c_mu_;
    double chiN_;

    // For sampling
    Eigen::MatrixXd B_; // eigenvectors of C
    Eigen::VectorXd D_; // eigenvalues of C

    std::mt19937                     rand_generator_{std::random_device{}()};
    std::normal_distribution<double> norm_dist_{0.0, 1.0};
};