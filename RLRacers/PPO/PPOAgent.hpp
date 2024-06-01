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
namespace rl
{
class PPOAgent : public Agent
{
  public:
    static constexpr float kVelocity{60.0};
    static constexpr float kSteeringDelta{5}; // degrees

    static constexpr float kClip{0.2};
    static constexpr float kLearningRate{0.01}; // learning rate for the Adam optimizer

    static constexpr size_t kStateDim{5};
    static constexpr int    kEpochPerEpisode{3};
    static constexpr int    kBatchSize{32};

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{{0, {kVelocity, 0.F}},
                                                                          {1, {kVelocity / 2.F, kSteeringDelta}},
                                                                          {2, {kVelocity / 2.F, -kSteeringDelta}}};

    struct ExperienceBuffer
    {
        std::vector<torch::Tensor> saved_log_probs;
        std::vector<torch::Tensor> saved_states;
        std::vector<float>         saved_rewards;
        std::vector<size_t>        saved_actions;

        torch::Tensor disc_rewards_tensor;

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(const size_t batch_size,
                                                                                      const size_t index_offset)
        {
            assert(disc_rewards_tensor.numel() != 0);

            std::vector<size_t>        actions_batch(batch_size);
            std::vector<torch::Tensor> log_probs_batch(batch_size);
            std::vector<torch::Tensor> states_batch(batch_size);
            size_t                     buffer_idx{index_offset};
            for (size_t i = 0; i < batch_size; ++i)
            {
                actions_batch[i]   = saved_actions[buffer_idx];
                log_probs_batch[i] = saved_log_probs[buffer_idx];
                states_batch[i]    = saved_states[buffer_idx];
                buffer_idx++;
            }
            torch::Tensor actions_tensor = torch::from_blob(actions_batch.data(),
                                                            {static_cast<int64_t>(batch_size)},
                                                            torch::TensorOptions().dtype(torch::kInt64))
                                               .clone()
                                               .unsqueeze(1);
            torch::Tensor log_probs_tensor = torch::stack(log_probs_batch, 0).unsqueeze(1);
            torch::Tensor states_tensor    = torch::stack(states_batch, 0);

            return std::make_tuple(disc_rewards_tensor.slice(0, index_offset, index_offset + batch_size).unsqueeze(1),
                                   actions_tensor,
                                   log_probs_tensor,
                                   states_tensor);
        }

        void calculateDiscountedRewards()
        {
            static constexpr float kGamma{0.99}; // discount factor btw current and future rewards
            constexpr float        kEps = std::numeric_limits<float>::epsilon();

            assert(disc_rewards_tensor.numel() == 0);

            std::vector<float> discounted_rewards(saved_rewards.size());
            float              cumulative_discounted_reward{0.F};

            // Reverse-iterate through rewards and calculate discounted returns
            size_t i = saved_rewards.size() - 1;
            for (auto r = saved_rewards.rbegin(); r != saved_rewards.rend(); ++r)
            {
                cumulative_discounted_reward = *r + kGamma * cumulative_discounted_reward;
                discounted_rewards[i]        = cumulative_discounted_reward;
                i--;
            }

            // Normalize the discounted rewards
            disc_rewards_tensor = torch::tensor(discounted_rewards);
            disc_rewards_tensor =
                (disc_rewards_tensor - disc_rewards_tensor.mean()) / (disc_rewards_tensor.std() + kEps).clone();
        }

        void clear()
        {
            saved_log_probs.clear();
            saved_states.clear();
            saved_rewards.clear();
            saved_actions.clear();

            disc_rewards_tensor = torch::Tensor();
        }
    };

    PPOAgent() = default;

    // Used when all agents are created initially, with randomized weights
    PPOAgent(const raylib::Vector2 start_pos,
             const float           start_rot,
             const int16_t         id,
             const size_t          start_idx     = 0,
             const size_t          track_idx_len = 0)
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
            int buffer_size = static_cast<int>(experience_buffer_.saved_actions.size());
            // We'll have some leftover samples that doesnt fit kBatchSize multiples
            for (auto idx_offset{0}; idx_offset < (buffer_size - kBatchSize); idx_offset += kBatchSize)
            {
                // Sample
                auto [disc_rewards_samples, actions_samples, log_probs_samples, states_samples] =
                    experience_buffer_.sample(kBatchSize, idx_offset);

                auto values = critic_.forward(states_samples);

                // advantages is related to the actor, hence we detach the values coming from the critic network to decouple their training
                torch::Tensor advantages = disc_rewards_samples - values.detach();

                // Generate new log probabilities with the updated actor, on the recorded states
                auto new_log_probs_all = torch::log(actor_.forward(states_samples));
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

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot, const size_t track_reset_idx)
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
};

PPOAgent::ExperienceBuffer PPOAgent::experience_buffer_;
Actor                      PPOAgent::actor_;
Critic                     PPOAgent::critic_;
torch::optim::Adam         PPOAgent::actor_optimizer_{
    torch::optim::Adam(PPOAgent::actor_.parameters(), PPOAgent::kLearningRate)};
torch::optim::Adam PPOAgent::critic_optimizer_{
    torch::optim::Adam(PPOAgent::critic_.parameters(), PPOAgent::kLearningRate)};

} // namespace rl