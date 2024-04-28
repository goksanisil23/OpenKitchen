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

#include "Network.hpp"
#include "ReplayBuffer.hpp"

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"

namespace rl
{
class DQLearnAgent : public Agent
{
  public:
    static constexpr float kVelocity{60.0};
    static constexpr float kSteeringDelta{5}; // degrees

    static constexpr float kEpsilon{0.9};          // probability of choosing random action (initial value)
    static constexpr float kEpsilonDiscount{0.01}; // how much epsilon decreases to next episode
    static constexpr float kGamma{0.8};            // discount factor btw current and future rewards
    static constexpr float kLearningRate{0.001};   // learning rate for the Adam optimizer

    static constexpr float kInvalidQVal{std::numeric_limits<float>::lowest()};

    static constexpr float kDangerProximity{5.0F};

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{{0, {kVelocity, 0.F}},
                                                                          {1, {kVelocity / 2.F, kSteeringDelta}},
                                                                          {2, {kVelocity / 2.F, -kSteeringDelta}}};

    DQLearnAgent() = default;

    // Used when all agents are created initially, with randomized weights
    DQLearnAgent(const raylib::Vector2 start_pos,
                 const float           start_rot,
                 const int16_t         id,
                 const size_t          start_idx     = 0,
                 const size_t          track_idx_len = 0)
        : Agent(start_pos, start_rot, id), prev_track_idx_{static_cast<int64_t>(start_idx)},
          track_idx_len_{static_cast<int64_t>(track_idx_len)}
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
    // decide on the next action based on epsilon-greedy policy
    void updateAction() override
    {
        torch::NoGradGuard no_grad;

        const float rand_val = static_cast<float>(GetRandomValue(0, RAND_MAX)) / static_cast<float>(RAND_MAX);
        if (rand_val < epsilon_)
        {
            current_action_idx_ = static_cast<size_t>(GetRandomValue(0, Network::kOutputSize - 1));
        }
        else
        {
            auto q_vals_for_this_state = nn_.forward(current_state_tensor_);
            // Choose the action corresponding to maximum q-value estimate
            current_action_idx_ = q_vals_for_this_state.argmax().item<int64_t>();
        }

        auto const accel_steer_pair{kActionMap.at(current_action_idx_)};
        current_action_.throttle_delta = accel_steer_pair.first;
        current_action_.steering_delta = accel_steer_pair.second;
    }

    void learn(const torch::Tensor &current_state_tensor,
               const size_t         current_action_idx,
               const float          reward,
               const torch::Tensor &next_state_tensor)
    {
        // Q-learning:
        // Q_new(s, a) = (1-α) * Q_current + α*(reward(s,a) + γ*max(Q(s'))

        auto q_values      = nn_.forward(current_state_tensor);
        auto next_q_values = nn_.forward(next_state_tensor);

        float temporal_diff_target = reward + kGamma * torch::max(next_q_values).item<float>();

        // Create a tensor identical to current q-value prediction
        auto target = q_values.clone().detach();
        // But just change 1 value which is our observation based q-value associated to next_action
        target[current_action_idx] = temporal_diff_target;

        auto loss = torch::mse_loss(q_values, target);
        optimizer_.zero_grad(); // clear the previous gradients
        loss.backward();
        optimizer_.step();
    }

    void updateDQN()
    {
        static constexpr int16_t kBatchSize{128};

        std::cout << "Replay buffer size: " << replay_buffer_.states.size() << std::endl;
        for (size_t iter{0}; iter < num_update_steps_; iter++)
        {
            // Sample replay buffer
            ReplayBuffer<size_t>::Samples samples = replay_buffer_.sample(kBatchSize);

            torch::Tensor state      = samples.states;
            torch::Tensor next_state = samples.next_states;
            torch::Tensor reward     = samples.rewards;
            torch::Tensor action     = samples.actions;

            auto q_values      = nn_.forward(state);
            auto next_q_values = nn_.forward(next_state);

            auto temporal_diff_target = reward + kGamma * torch::max(next_q_values).item<float>();

            // Correct way to use action indices to select corresponding q-values for updating
            auto gather_index     = action.unsqueeze(-1); // Ensure it has the correct shape for gather
            auto q_value_selected = q_values.gather(1, gather_index).squeeze(1);

            // Create a target tensor
            auto target = q_values.clone().detach();
            target.scatter_(1, gather_index, temporal_diff_target.unsqueeze(1));

            auto loss = torch::mse_loss(q_values, target);
            optimizer_.zero_grad(); // clear the previous gradients
            loss.backward();
            optimizer_.step();
        }
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot, const size_t track_reset_idx)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());

        prev_track_idx_ = track_reset_idx;
    }

    // Calculates the current reward of the agent
    float reward(const size_t nearest_track_idx)
    {
        if (crashed_)
        {
            return -200.F;
        }

        // Progression based reward
        float   progression_reward{0.F};
        int64_t progression = nearest_track_idx - prev_track_idx_;

        prev_track_idx_ = nearest_track_idx;

        // Handle wrap around: progression can't be more than half the track length
        progression_reward = std::abs(progression) > (track_idx_len_ / 2) ? track_idx_len_ - std::abs(progression)
                                                                          : std::abs(progression);

        return progression_reward * 10.F;
    }

    std::vector<float> getCurrentState() const
    {
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return sensor_hits_norm;
    }

  public:
    size_t        current_action_idx_;
    torch::Tensor current_state_tensor_;

    // Shared across agents
    static Network            nn_;
    static torch::optim::Adam optimizer_;

    int64_t prev_track_idx_;
    int64_t track_idx_len_;

    float epsilon_{kEpsilon};

    size_t               num_update_steps_{200};
    ReplayBuffer<size_t> replay_buffer_;
};

Network            DQLearnAgent::nn_        = Network();
torch::optim::Adam DQLearnAgent::optimizer_ = torch::optim::Adam(nn_.parameters(), kLearningRate);

} // namespace rl