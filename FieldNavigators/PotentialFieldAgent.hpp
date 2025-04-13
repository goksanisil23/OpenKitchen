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
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"

// Potential Field based navigation agent
class PotFieldAgent : public Agent
{
  public:
    static constexpr size_t kLookAheadIdx        = 2; // race track index lookahead
    static constexpr float  kAttractiveConstant  = 100.0f;
    static constexpr float  kRepulsiveConstant   = 10.0f;
    static constexpr float  kObstacleEffectRange = 5.0f; // Range within which obstacles exert force

    PotFieldAgent() = default;

    // Used when all agents are created initially, with randomized weights
    PotFieldAgent(const Vec2d   start_pos,
                  const float   start_rot,
                  const int16_t id,
                  const size_t  start_idx     = 0,
                  const size_t  track_idx_len = 0)
        : Agent(start_pos, start_rot, id)
    {
        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-90.F);
        sensor_ray_angles_.push_back(-60.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(60.F);
        sensor_ray_angles_.push_back(90.F);
    }

    void setGoalPoint(const Vec2d goal)
    {
        goal_point_ = goal;
    }

    void updateAction()
    {
        Vec2d attractive_force = goal_point_ - this->pos_;
        // Normalize the attractive force
        float distance_to_goal = attractive_force.length();
        attractive_force       = attractive_force / distance_to_goal * kAttractiveConstant;

        Vec2d repulsive_force;
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {

            if (sensor_hits_[i].norm() < kObstacleEffectRange)
            {
                float angle = sensor_ray_angles_[i];
                // COmpute force vector for this obstacle
                float force_magnitude =
                    kRepulsiveConstant * (1.f / sensor_hits_[i].norm() - 1.f / kObstacleEffectRange);
                repulsive_force.x += std::cos(angle * M_PI / 180.f) * force_magnitude;
                repulsive_force.y += std::sin(angle * M_PI / 180.f) * force_magnitude;
            }
        }

        Vec2d total_force = attractive_force - repulsive_force;

        // Determine control based on forces
        auto goal_rotation = std::atan2(total_force.y, total_force.x) * 180.F / M_PI;
        // Bound to [0,100]
        current_action_.throttle_delta = std::min(total_force.length(), 100.F); // goal speed
        current_action_.steering_delta = goal_rotation - rot_;
        // Bound to -180,180
        current_action_.steering_delta = normalizeAngleDeg(current_action_.steering_delta);
        if (current_action_.steering_delta > 180.F)
            current_action_.steering_delta -= 360.F;
    }

    void reset(const Vec2d &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

  private:
    Vec2d goal_point_;
};