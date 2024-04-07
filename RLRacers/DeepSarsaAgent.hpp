#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"
#include "Network.hpp"

namespace deep_sarsa
{
class DeepSarsaAgent : public Agent
{
  public:
    static constexpr float kAccelerationDelta{1.0};
    static constexpr float kSteeringDeltaLow{0.2}; // degrees
    static constexpr float kSteeringDeltaHigh{0.8};

    static constexpr float kEpsilon{0.7};        // probability of choosing random action
    static constexpr float kGamma{0.9};          // discount factor btw current and future rewards
    static constexpr float kLearningRate{0.001}; // learning rate for the Adam optimizer

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{
        {0, {kAccelerationDelta, 0.F}},
        {1, {-kAccelerationDelta, 0.F}},
        {2, {kAccelerationDelta, kSteeringDeltaLow}},
        {3, {kAccelerationDelta, kSteeringDeltaHigh}},
        {4, {kAccelerationDelta, -kSteeringDeltaLow}},
        {5, {kAccelerationDelta, -kSteeringDeltaHigh}},
        {6, {-kAccelerationDelta, kSteeringDeltaLow}},
        {7, {-kAccelerationDelta, kSteeringDeltaHigh}},
        {8, {-kAccelerationDelta, -kSteeringDeltaLow}},
        {9, {-kAccelerationDelta, -kSteeringDeltaHigh}}};

    DeepSarsaAgent() = default;

    // Used when all agents are created initially, with randomized weights
    DeepSarsaAgent(raylib::Vector2 start_pos, float start_rot, int16_t id) : Agent(start_pos, start_rot, id)
    {
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        // populate the input, but normalize the features so that all lie in [0,1]
        auto const         speed_norm = speed_ / kSpeedLimit; // since we don't allow (-) speed
        auto const         rot_norm   = normalizeAngleDeg(rot_) / kRotationLimit;
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return torch::cat({torch::tensor({speed_norm}), torch::tensor({rot_norm}), torch::tensor(sensor_hits_norm)});
    }

    // Given the current state of the agent (measurements + internal states),
    // decide on the next action based on epsilon-greedy policy
    void updateAction() override
    {
        torch::NoGradGuard no_grad;

        if (torch::rand(1).item<float>() < kEpsilon)
        {
            next_action_idx_ = torch::randint(0, Network::kOutputSize, {1}).item<int32_t>();
        }
        else
        {
            next_state_tensor_ = stateToTensor();
            auto next_q_values = nn_.forward(next_state_tensor_);
            // Choose the action corresponding to maximum q-value estimate
            next_action_idx_ = next_q_values.argmax().item<int64_t>();
        }

        auto const accel_steer_pair{kActionMap.at(next_action_idx_)};
        next_action_.throttle_delta = accel_steer_pair.first;
        next_action_.steering_delta = accel_steer_pair.second;
    }

    void train(const float reward, const bool done)
    {
        auto q_values      = nn_.forward(state_tensor_);
        auto next_q_values = nn_.forward(next_state_tensor_);

        float target_val = reward;
        if (!done)
        {
            target_val += kGamma * next_q_values[next_action_idx_].item<float>();
        }
        // Create a tensor identical to current q-value prediction
        auto target = q_values.clone().detach();
        // But just change 1 value which is our observation based q-value associated to next_action
        target[current_action_idx_] = target_val;

        auto loss = torch::mse_loss(q_values, target);
        optimizer_.zero_grad(); // clear the previous gradients
        loss.backward();
        optimizer_.step();
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());

        next_action_        = Action();
        next_action_idx_    = -1;
        current_action_idx_ = -1;
        state_tensor_       = stateToTensor();
        next_state_tensor_  = stateToTensor();
    }

    // Calculates the current rewards of the agent
    float reward() const
    {
        if (crashed_)
        {
            return -10.F;
        }
        float avg_sensor_dist{0.F};
        for (auto const hit : sensor_hits_)
        {
            avg_sensor_dist += hit.norm();
        }
        // float sensor_reward = avg_sensor_dist / (static_cast<float>(sensor_hits_.size()) * kSensorRange); // [0,1]
        float sensor_reward = 0;

        float speed_reward = speed_ * 5.F / kSpeedLimit; //[0,1]

        // return sensor_reward;
        return sensor_reward + speed_reward;
        // return static_cast<float>(race_track_->findNearestTrackIndexBruteForce({agent_->pos_.x, agent_->pos_.y}));
    }

  public:
    Action  next_action_;
    int64_t next_action_idx_;
    int64_t current_action_idx_;

    // Shared across agents
    static Network            nn_;
    static torch::optim::Adam optimizer_;

    torch::Tensor state_tensor_;
    torch::Tensor next_state_tensor_;
};

Network            DeepSarsaAgent::nn_        = Network();
torch::optim::Adam DeepSarsaAgent::optimizer_ = torch::optim::Adam(nn_.parameters(), kLearningRate);

} // namespace deep_sarsa