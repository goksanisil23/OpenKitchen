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

namespace rl
{
class QLearnAgent : public Agent
{
  public:
    static constexpr float kVelocity{60.0};
    static constexpr float kSteeringDelta{5}; // degrees

    static constexpr float kEpsilon{0.9};          // probability of choosing random action (initial value)
    static constexpr float kEpsilonDiscount{0.05}; // how much epsilon decreases to next episode
    static constexpr float kGamma{0.8};            // discount factor btw current and future rewards
    static constexpr float kAlpha{0.2};

    // 5 rays, 3 regions based on proximity --> state
    // 3 actions per state
    static constexpr size_t   kActionSize{3};
    static constexpr uint32_t kNumStates{243};

    static constexpr float kInvalidQVal{std::numeric_limits<float>::lowest()};

    static constexpr float kDangerProximity{5.0F};

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{{0, {kVelocity, 0.F}},
                                                                          {1, {kVelocity / 2.F, kSteeringDelta}},
                                                                          {2, {kVelocity / 2.F, -kSteeringDelta}}};

    QLearnAgent() = default;

    // Used when all agents are created initially, with randomized weights
    QLearnAgent(const raylib::Vector2 start_pos,
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

        // Set all the Q-values to invalid
        for (auto &action_vals_per_state : q_table_)
        {
            std::fill(action_vals_per_state.begin(), action_vals_per_state.end(), kInvalidQVal);
        }
    }

    // Discretizes the sensor readings into binned regions
    size_t discretizeState()
    {
        size_t state_index = 0;
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            size_t     bin;
            auto const ray_dist = sensor_hits_[i].norm();
            if (ray_dist < kDangerProximity)
            {
                bin = 0;
            }
            else if (ray_dist < (kDangerProximity * 2.F))
            {
                bin = 1;
            }
            else
            {
                bin = 2;
            }
            state_index += bin * std::pow(3, i);
        }
        return state_index;
    }

    // Given the current state of the agent (measurements + internal states),
    // decide on the next action based on epsilon-greedy policy
    void updateAction() override
    {
        const float rand_val = static_cast<float>(GetRandomValue(0, RAND_MAX)) / static_cast<float>(RAND_MAX);
        if (rand_val < epsilon_)
        {
            current_action_idx_ = static_cast<size_t>(GetRandomValue(0, kActionSize - 1));
        }
        else
        {
            auto   q_vals_for_this_state = q_table_[current_state_idx_];
            size_t max_val_action =
                std::distance(q_vals_for_this_state.begin(),
                              std::max_element(q_vals_for_this_state.begin(), q_vals_for_this_state.end()));

            // Choose the action corresponding to maximum q-value estimate
            current_action_idx_ = max_val_action;
        }

        auto const accel_steer_pair{kActionMap.at(current_action_idx_)};
        current_action_.throttle_delta = accel_steer_pair.first;
        current_action_.steering_delta = accel_steer_pair.second;
    }

    void learn(const size_t current_state_idx, const size_t action, const float reward, const size_t next_state_idx)
    {
        // Q-learning:
        // Q_new(s, a) = (1-α) * Q_current + α*(reward(s,a) + γ*max(Q(s'))

        float max_q_next_state = *std::max_element(q_table_[next_state_idx].begin(), q_table_[next_state_idx].end());
        float temporal_diff_target = reward + kGamma * max_q_next_state;

        float old_q = q_table_.at(current_state_idx_).at(current_action_idx_);
        if ((old_q == kInvalidQVal) || (max_q_next_state == kInvalidQVal))
        {
            q_table_.at(current_state_idx_).at(current_action_idx_) = reward;
        }
        else
        {
            q_table_.at(current_state_idx_).at(current_action_idx_) = old_q + kAlpha * (temporal_diff_target - old_q);
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

        return progression_reward;
    }

  public:
    size_t current_action_idx_;
    size_t current_state_idx_;

    int64_t prev_track_idx_;
    int64_t track_idx_len_;

    float epsilon_{kEpsilon};
    float score_{0.F};

    std::array<std::array<float, kActionSize>, kNumStates> q_table_;
};

} // namespace rl