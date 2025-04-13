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
    static constexpr float kVelocity{60.0};
    static constexpr float kSteeringDelta{5}; // degrees

    static constexpr float kGamma{0.99};        // discount factor btw current and future rewards
    static constexpr float kLearningRate{0.01}; // learning rate for the Adam optimizer

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{{0, {kVelocity, 0.F}},
                                                                          {1, {kVelocity / 2.F, kSteeringDelta}},
                                                                          {2, {kVelocity / 2.F, -kSteeringDelta}}};

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
        // populate the input, but normalize the features so that all lie in [0,1]
        // auto const         speed_norm = speed_ / kSpeedLimit; // since we don't allow (-) speed
        // auto const         rot_norm   = normalizeAngleDeg(rot_) / kRotationLimit;
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        // return torch::cat({torch::tensor({speed_norm}), torch::tensor({rot_norm}), torch::tensor(sensor_hits_norm)});
        return torch::tensor(sensor_hits_norm);
    }

    // Given the current state of the agent (measurements + internal states),
    // decide on the next action based on policy network
    void updateAction() override
    {
        // Probability associated to each action
        auto action_probs = policy_.forward(current_state_tensor_.unsqueeze(0));
        // Given action probabilities per each action index, sample an action index
        // This is more relaxed than greedy policy, still allowing some randomness.
        torch::Tensor sampled_action = torch::multinomial(action_probs, /*num_samples=*/1);
        // Calculate the log probability of the chosen action
        torch::Tensor log_prob = torch::log(action_probs.index({0, sampled_action.item<int>()}));
        // Save the log probability
        policy_.saved_log_probs.push_back(log_prob);

        // Choose the action corresponding to maximum q-value estimate
        current_action_idx_ = sampled_action.item<int>();

        auto const accel_steer_pair{kActionMap.at(current_action_idx_)};
        current_action_.throttle_delta = accel_steer_pair.first;
        current_action_.steering_delta = accel_steer_pair.second;
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

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot, const size_t track_reset_idx)
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