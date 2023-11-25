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

    static constexpr float kEpsilon{0.9}; // probability of choosing random action (initial value)
    // static constexpr float kEpsilonDiscount{0.9986}; // how much epsilon decreases to next episode
    static constexpr float kEpsilonDiscount{0.92};
    static constexpr float kGamma{0.8}; // discount factor btw current and future rewards
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
    QLearnAgent(raylib::Vector2 start_pos, float start_rot, int16_t id) : Agent(start_pos, start_rot, id)
    {

        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-70.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(70.F);

        // Set all the Q-values to invalid
        for (auto &action_vals_per_state : q_values_)
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
            current_action_idx_ = static_cast<size_t>(GetRandomValue(0, 2));
        }
        else
        {
            auto   q_vals_for_this_state = q_values_[current_state_idx_];
            size_t max_val_action =
                std::distance(q_vals_for_this_state.begin(),
                              std::max_element(q_vals_for_this_state.begin(), q_vals_for_this_state.end()));

            // Choose the action corresponding to maximum q-value estimate
            current_action_idx_ = max_val_action;
        }

        auto const accel_steer_pair{kActionMap.at(current_action_idx_)};
        current_action_.acceleration_delta = accel_steer_pair.first;
        current_action_.steering_delta     = accel_steer_pair.second;
    }

    void learn(const size_t current_state_idx, const size_t action, const float reward, const size_t next_state_idx)
    {
        // Q-learning:
        // Q_new(s, a) = (1-α) * Q_current + α*(reward(s,a) + γ*max(Q(s'))

        float max_q_next_state = *std::max_element(q_values_[next_state_idx].begin(), q_values_[next_state_idx].end());
        float temporal_diff_target = reward + kGamma * max_q_next_state;

        float old_q = q_values_.at(current_state_idx_).at(current_action_idx_);
        if ((old_q == kInvalidQVal) || (max_q_next_state == kInvalidQVal))
        {
            q_values_.at(current_state_idx_).at(current_action_idx_) = reward;
        }
        else
        {
            q_values_.at(current_state_idx_).at(current_action_idx_) = old_q + kAlpha * (temporal_diff_target - old_q);
        }
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

    // Calculates the current rewards of the agent
    float reward(const size_t nearest_track_idx) const
    {
        if (crashed_)
        {
            return -200.F;
        }

        float  dist_based_reward{0.F};
        size_t danger_ctr = 0;
        // Penalize by the amount of rays reporting dangerously close
        for (auto const hit : sensor_hits_)
        {
            auto const ray_dist = hit.norm();
            if (ray_dist < kDangerProximity)
            {
                danger_ctr++;
            }
        }
        if (danger_ctr >= 3)
            dist_based_reward = -50;
        else if (danger_ctr == 2)
            dist_based_reward = 0;
        else
            dist_based_reward = 5;

        // Progression based reward
        float          progression_award{0.F};
        static int64_t prev_track_idx{static_cast<int64_t>(nearest_track_idx)};
        int64_t        progression = nearest_track_idx - prev_track_idx;
        if (progression > 0)
            progression_award = 20.F;
        else
            progression_award = -20.F;
        prev_track_idx = nearest_track_idx;

        return dist_based_reward + progression_award;
    }

  public:
    size_t current_action_idx_;
    size_t current_state_idx_;

    float epsilon_{kEpsilon};

    std::array<std::array<float, kActionSize>, kNumStates> q_values_;
};

} // namespace rl