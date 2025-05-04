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

#include "Actor.hpp"
#include "Critic.hpp"
#include "ReplayBuffer.hpp"

#include "Environment/Agent.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"

namespace rl
{
class DDPGAgent : public Agent
{
  public:
    static constexpr float kGamma{0.99};              // discount factor btw current and future rewards
    static constexpr float kActorLearningRate{1e-4};  // learning rate for Actor's the Adam optimizer
    static constexpr float kCriticLearningRate{1e-3}; // learning rate for Actor's the Adam optimizer
    static constexpr float kTau{0.005};               // target smoothing factor (weight given to target's network)

    static constexpr size_t                                             kStateDim{5};
    static constexpr size_t                                             kActionDim{2};
    typedef ReplayBuffer<kStateDim, kActionDim, float, torch::kFloat32> ReplayBufferDDPG;

    DDPGAgent() = default;

    // Used when all agents are created initially, with randomized weights
    DDPGAgent(const Vec2d   start_pos,
              const float   start_rot,
              const int16_t id,
              const size_t  start_idx     = 0,
              const size_t  track_idx_len = 0)
        : Agent(start_pos, start_rot, id),
          actor_optimizer_{torch::optim::Adam(actor_.parameters(), kActorLearningRate)},
          critic_optimizer_{torch::optim::Adam(critic_.parameters(), kCriticLearningRate)}
    {

        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-70.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(70.F);

        assert(sensor_ray_angles_.size() == kStateDim);

        // Target networks should start with equal weights
        torch::NoGradGuard no_grad;
        for (const auto &key_value_pair : critic_.named_parameters())
        {
            critic_target_.named_parameters()[key_value_pair.key()].copy_(key_value_pair.value());
        }
        for (const auto &key_value_pair : actor_.named_parameters())
        {
            actor_target_.named_parameters()[key_value_pair.key()].copy_(key_value_pair.value());
        }
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
    // decide on the next action based on actor network
    void updateAction() override
    {
        // Probability associated to each action
        auto action = actor_.forward(current_state_tensor_.unsqueeze(0));

        current_action_.throttle_delta = action[0][0].item<float>();
        current_action_.steering_delta = action[0][1].item<float>();
    }

    template <typename T>
    static void updateTargetNetwork(const T &source_network, T &target_network)
    {
        auto source_params = source_network.named_parameters(true); // recurse
        auto target_params = target_network.named_parameters(true); // recurse

        for (const auto &source_param : source_params)
        {
            auto &target_param = target_params[source_param.key()];
            target_param.set_data(kTau * source_param.value() + (1 - kTau) * target_param);
        }
    }

    void updateDDPG()
    {
        static constexpr int16_t kBatchSize{250};

        std::cout << "Replay buffer size: " << replay_buffer_.states.size() << std::endl;
        for (size_t iter{0}; iter < num_update_steps_; iter++)
        {
            // Sample replay buffer
            ReplayBufferDDPG::Samples samples = replay_buffer_.sample(kBatchSize);

            torch::Tensor state      = samples.states;
            torch::Tensor next_state = samples.next_states;
            torch::Tensor reward     = samples.rewards;
            torch::Tensor action     = samples.actions;
            torch::Tensor done       = samples.dones;

            // Compute the target Q value, using target networks only
            auto target_Q = critic_target_.forward(next_state, actor_target_.forward(next_state));
            target_Q      = reward + ((1 - done) * kGamma * target_Q).detach();

            // Get the curent Q estimate
            auto current_Q = critic_.forward(state, action);

            // Compute Critic's loss
            auto critic_loss = torch::mse_loss(current_Q, target_Q);

            // Optimize Critic
            critic_optimizer_.zero_grad();
            critic_loss.backward();
            critic_optimizer_.step();

            // Compute Actor's loss
            auto actor_loss = -critic_.forward(state, actor_.forward(state)).mean();

            // Optimize Actor
            actor_optimizer_.zero_grad();
            actor_loss.backward();
            actor_optimizer_.step();

            // Update the frozen target models
            updateTargetNetwork(actor_, actor_target_);
            updateTargetNetwork(critic_, critic_target_);
        }
    }

    void reset(const Vec2d &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
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

    std::array<float, kActionDim> getCurrentAction() const
    {
        return {current_action_.throttle_delta, current_action_.steering_delta};
    }

  public:
    torch::Tensor current_state_tensor_;

    Actor  actor_{};
    Critic critic_{};

    Actor  actor_target_{};
    Critic critic_target_{};

    torch::optim::Adam actor_optimizer_;
    torch::optim::Adam critic_optimizer_;

    size_t num_update_steps_{50};

    ReplayBufferDDPG replay_buffer_; // specify state and action dimensions
};

} // namespace rl