#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"
#include "Policy.hpp"

namespace rl
{
class ReinforceAgent : public Agent
{
  public:
    static constexpr float kGamma{0.99};         // discount factor btw current and future rewards
    static constexpr float kLearningRate{0.001}; // learning rate for the Adam optimizer

    ReinforceAgent() = default;

    // Used when all agents are created initially, with randomized weights
    ReinforceAgent(const Vec2d   start_pos,
                   const float   start_rot,
                   const int16_t id,
                   const size_t  start_idx     = 0,
                   const size_t  track_idx_len = 0)
        : Agent(start_pos, start_rot, id), optimizer_{torch::optim::Adam(policy_.parameters(), kLearningRate)}
    {

        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-70.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(70.F);
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return torch::tensor(sensor_hits_norm);
    }

    // Given the current state of the agent (measurements + internal states),
    // decide on the next action based on policy network
    void updateAction() override
    {
        auto [mu, log_std] = policy_.forward(current_state_tensor_.unsqueeze(0));
        // std::cout << "mu: " << mu[0][0].item<float>() << " " << mu[0][1].item<float>() << std::endl;
        std::cout << "log_std: " << log_std[0][0].item<float>() << " " << log_std[0][1].item<float>() << std::endl;

        torch::Tensor std = torch::exp(log_std);

        // Manual normal distribution sampling
        torch::Tensor action_pre_tanh = mu + std * torch::randn_like(mu);
        torch::Tensor action          = torch::tanh(action_pre_tanh);

        // Manual log probability calculation
        torch::Tensor log_prob =
            (-0.5 * ((action_pre_tanh - mu) / std).pow(2) - torch::log(std) - 0.5 * std::log(2 * M_PI))
                .sum(-1); // Sum over action dimensions

        // Tanh correction (log derivative)
        log_prob -= torch::log(1 - action.pow(2) + 1e-6).sum(-1);
        policy_.saved_log_probs.push_back(log_prob.squeeze(0).clone());

        auto const action_out          = action.detach();
        current_action_.throttle_delta = ((action_out[0][0].item<float>() + 1.F) * 0.5F) * 100.F;
        current_action_.steering_delta = (action_out[0][1].item<float>()) * 10.F;
    }

    void updatePolicy()
    {
        constexpr float   kEps = std::numeric_limits<float>::epsilon();
        float             cumulative_discounted_reward{0.F};
        std::deque<float> returns;

        // Reverse-iterate through rewards and calculate discounted returns
        for (auto r = policy_.rewards.rbegin(); r != policy_.rewards.rend(); ++r)
        {
            cumulative_discounted_reward = *r + kGamma * cumulative_discounted_reward;
            returns.push_front(cumulative_discounted_reward);
        }

        // Convert returns to a tensor and normalize
        torch::Tensor returns_tensor = torch::tensor(std::vector<float>(returns.begin(), returns.end()));
        returns_tensor               = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + kEps);

        // Calculate policy loss
        torch::Tensor loss = torch::tensor(0.0);
        for (size_t i = 0; i < policy_.saved_log_probs.size(); ++i)
        {
            loss += (-policy_.saved_log_probs[i]) * returns_tensor[i];
        }

        // Perform backpropagation
        optimizer_.zero_grad();
        loss.backward();
        optimizer_.step();

        // Clear the rewards and saved log probabilities
        policy_.rewards.clear();
        policy_.saved_log_probs.clear();
    }

    void reset(const Vec2d &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

  public:
    size_t        current_action_idx_;
    torch::Tensor current_state_tensor_;

    Policy             policy_{};
    torch::optim::Adam optimizer_;
};

} // namespace rl