#pragma once

#undef NDEBUG

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

    static constexpr float kEpsilon{0.99};         // probability of choosing random action (initial value)
    static constexpr float kEpsilonDiscount{0.01}; // how much epsilon decreases to next episode
    static constexpr float kGamma{0.99};           // discount factor btw current and future rewards
    static constexpr float kLearningRate{1e-4};    // learning rate for the Adam optimizer

    static constexpr size_t                                             kStateDim{5};
    static constexpr size_t                                             kActionDim{1}; // since we only choose index
    typedef ReplayBuffer<kStateDim, kActionDim, int64_t, torch::kInt64> ReplayBufferDQN;

    const std::unordered_map<int64_t, std::pair<float, float>> kActionMap{
        {0, {kVelocity, 0.F}},
        {1, {kVelocity / 2.F, kSteeringDelta}},
        {2, {kVelocity / 2.F, -kSteeringDelta}},
        {3, {kVelocity / 2.F, kSteeringDelta / 2.F}},
        {4, {kVelocity / 2.F, -kSteeringDelta / 2.F}}};

    DQLearnAgent() = default;

    // Used when all agents are created initially, with randomized weights
    DQLearnAgent(const Vec2d   start_pos,
                 const float   start_rot,
                 const int16_t id,
                 const size_t  start_idx     = 0,
                 const size_t  track_idx_len = 0)
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

        assert(sensor_ray_angles_.size() == kStateDim);
        assert(kActionMap.size() == Network::kOutputSize);
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        // populate the input, but normalize the features so that all lie in [0,1]
        std::vector<float> sensor_hits_normalized(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_normalized[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return torch::tensor(sensor_hits_normalized);
    }

    // Given the current state of the agent (measurements + internal states),
    // decide on the next action based on epsilon-greedy policy
    void updateAction() override
    {
        torch::NoGradGuard no_grad;

        const float rand_val = static_cast<float>(GetRandomValue(0, RAND_MAX)) / static_cast<float>(RAND_MAX);
        if (rand_val < epsilon_)
        {
            current_action_idx_ = static_cast<int64_t>(GetRandomValue(0, Network::kOutputSize - 1));
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

    static void updateDQN()
    {
        static constexpr int16_t kBatchSize{100};
        static constexpr size_t  kIterationSteps{200};

        std::cout << "Replay buffer size: " << replay_buffer_.states.size() << std::endl;
        // Sample replay buffer
        DQLearnAgent::ReplayBufferDQN::Samples samples = replay_buffer_.sample(kBatchSize);
        for (size_t iter{0}; iter < kIterationSteps; iter++)
        {
            // Alternatively: Sample a replay buffer at each gradient descent step
            // ReplayBuffer<size_t>::Samples samples = replay_buffer_.sample(kBatchSize);

            torch::Tensor state        = samples.states;
            torch::Tensor next_state   = samples.next_states;
            torch::Tensor reward       = samples.rewards;
            torch::Tensor action_index = samples.actions;
            torch::Tensor done         = samples.dones;

            std::cout << "state: " << state.sizes() << std::endl;
            std::cout << "reward: " << reward.sizes() << std::endl;
            std::cout << "action_index: " << action_index.sizes() << std::endl;

            auto q_values      = nn_.forward(state);
            auto next_q_values = nn_.forward(next_state).detach();

            // Both targets below works
            auto temporal_diff_target = reward + kGamma * torch::amax(next_q_values, 1, true);
            // auto temporal_diff_target = (reward + (1 - done) * kGamma * torch::amax(next_q_values, 1, true)).detach();

            // Create a target tensor
            auto target = q_values.clone().detach();

            for (int batch_idx{0}; batch_idx < target.size(0); ++batch_idx)
            {
                target[batch_idx].index_put_({action_index[batch_idx].squeeze()},
                                             temporal_diff_target[batch_idx].squeeze());
            }

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
    float calculateReward(const size_t nearest_track_idx)
    {
        if (crashed_)
        {
            return -200.F;
        }

        float min_distance = Agent::kSensorRange;
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            if (min_distance > sensor_hits_[i].norm())
            {
                min_distance = sensor_hits_[i].norm();
            }
        }

        float progression_reward = min_distance;

        return progression_reward;
    }

    std::array<float, kStateDim> getCurrentState() const
    {
        std::array<float, kStateDim> sensor_hits_norm;
        for (size_t i{0}; i < kStateDim; i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return sensor_hits_norm;
    }

    std::array<int64_t, kActionDim> getCurrentAction() const
    {
        return {current_action_idx_};
    }

  public:
    int64_t       current_action_idx_;
    torch::Tensor current_state_tensor_;

    int64_t prev_track_idx_;
    int64_t track_idx_len_;

    float epsilon_{kEpsilon};

    // Variables shared among all agents
    static ReplayBufferDQN    replay_buffer_;
    static Network            nn_;
    static torch::optim::Adam optimizer_;
};

DQLearnAgent::ReplayBufferDQN DQLearnAgent::replay_buffer_;
Network                       DQLearnAgent::nn_{Network()};
torch::optim::Adam            DQLearnAgent::optimizer_{DQLearnAgent::nn_.parameters(), kLearningRate};

} // namespace rl