#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"

#include "Actor.hpp"
#include "Critic.hpp"
#include "ExperienceBuffer.hpp"
namespace rl
{

namespace
{
constexpr float kProbClamp{1e-8};
}
class PPOAgent : public Agent
{
  public:
    static constexpr float kVelocity{60.0};
    static constexpr float kSteeringDelta{5}; // degrees

    static constexpr float kClip{0.2};
    static constexpr float kLearningRate{3e-4}; // learning rate for the Adam optimizer

    static constexpr size_t kStateDim{5};
    static constexpr int    kEpochPerEpisode{5};
    static constexpr int    kBatchSize{64};

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{{0, {kVelocity, 0.F}},
                                                                          {1, {kVelocity / 2.F, kSteeringDelta}},
                                                                          {2, {kVelocity / 2.F, -kSteeringDelta}}};

    PPOAgent() = default;

    // Used when all agents are created initially, with randomized weights
    PPOAgent(const Vec2d   start_pos,
             const float   start_rot,
             const int16_t id,
             const size_t  start_idx     = 0,
             const size_t  track_idx_len = 0)
        : Agent(start_pos, start_rot, id)
    {

        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-70.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(70.F);

        assert(sensor_ray_angles_.size() == kStateDim);
    }

    std::vector<float> getCurrentState() const
    {
        std::vector<float> sensor_hits_norm(kStateDim);
        for (size_t i{0}; i < kStateDim; i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return sensor_hits_norm;
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        return torch::tensor(getCurrentState());
    }

    void updateAction()
    {
        // Probability associated to each action
        torch::Tensor action_probs = actor_.forward(current_state_tensor_.unsqueeze(0));
        action_probs               = torch::clamp(action_probs, kProbClamp, 1.0 - kProbClamp);
        // Given action probabilities per each action index, sample an action index
        // This is more relaxed than greedy policy, still allowing some randomness.
        torch::Tensor sampled_action = torch::multinomial(action_probs, /*num_samples=*/1);
        // Calculate the log probability of the chosen action
        torch::Tensor log_prob = torch::log(action_probs.index({0, sampled_action.item<int>()}));

        experience_buffer_.saved_log_probs.push_back(log_prob.detach());

        // Choose the action corresponding to maximum q-value estimate
        current_action_idx_ = sampled_action.item<int>();

        current_action_.throttle_delta = kActionMap.at(current_action_idx_).first;
        current_action_.steering_delta = kActionMap.at(current_action_idx_).second;
    }

    static void updatePolicy()
    {
        experience_buffer_.calculateDiscountedRewards();

        // Train on the entire replay buffer, in BATCH sizes
        for (auto i{0}; i < kEpochPerEpisode; i++)
        {
            const int buffer_size = static_cast<int>(experience_buffer_.saved_actions.size());
            // We'll have some leftover samples that doesnt fit kBatchSize multiples
            // for (auto idx_offset{0}; idx_offset < (buffer_size - kBatchSize); idx_offset += kBatchSize)
            for (int i = 0; i < buffer_size; i += kBatchSize)
            {
                const int batch_size = std::min(kBatchSize, buffer_size - i);
                // Sample
                auto [disc_rewards_samples, actions_samples, log_probs_samples, states_samples] =
                    // experience_buffer_.sample(kBatchSize, idx_offset);
                    experience_buffer_.sample(batch_size, i);

                auto values = critic_.forward(states_samples);

                // advantages is related to the actor, hence we detach the values coming from the critic network to decouple their training
                torch::Tensor advantages = disc_rewards_samples - values.detach();

                // Generate new log probabilities with the updated actor, on the recorded states
                auto new_probs_all     = torch::clamp(actor_.forward(states_samples), kProbClamp, 1.0 - kProbClamp);
                auto new_log_probs_all = torch::log(new_probs_all);
                // Select the log probabilities of the recorded actions
                auto new_action_log_probs = new_log_probs_all.gather(1, actions_samples);
                // Ratio represents difference between the old and new action probs
                torch::Tensor ratios = torch::exp(new_action_log_probs - log_probs_samples);

                // The clipping limits the effective change actor can make at each step in order to improve stability.
                // So at first, none of actor updates will be clipped and its guaranteed to learn something from these samples. But as actor changes along batches and epochs, saved vs newly generated action probabilities will diverge, hence ratio will grow
                torch::Tensor surr1      = ratios * advantages;
                torch::Tensor surr2      = torch::clamp(ratios, 1.0 - kClip, 1.0 + kClip) * advantages;
                torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
                // Note that values are not detached here since it needs to minimize the difference between predicted & actual discounted returns
                torch::Tensor critic_loss = torch::mse_loss(values, disc_rewards_samples);

                actor_optimizer_.zero_grad();
                actor_loss.backward();
                actor_optimizer_.step();

                critic_optimizer_.zero_grad();
                critic_loss.backward();
                critic_optimizer_.step();
            }
        }

        experience_buffer_.clear();
    }

    inline size_t getCurrentAction()
    {
        return current_action_idx_;
    }

    void reset(const Vec2d &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

  public:
    size_t        current_action_idx_;
    torch::Tensor current_state_tensor_;

    // Shared across all parallel agents
    static ExperienceBuffer experience_buffer_;

    static Actor              actor_;
    static Critic             critic_;
    static torch::optim::Adam actor_optimizer_;
    static torch::optim::Adam critic_optimizer_;

    size_t prev_track_idx_{};
};

ExperienceBuffer   PPOAgent::experience_buffer_;
Actor              PPOAgent::actor_;
Critic             PPOAgent::critic_;
torch::optim::Adam PPOAgent::actor_optimizer_{
    torch::optim::Adam(PPOAgent::actor_.parameters(), PPOAgent::kLearningRate)};
torch::optim::Adam PPOAgent::critic_optimizer_{
    torch::optim::Adam(PPOAgent::critic_.parameters(), PPOAgent::kLearningRate)};

} // namespace rl