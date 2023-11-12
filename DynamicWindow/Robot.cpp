#include "Robot.h"

#include "raylib-cpp.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
float closestDistToObstacle(const Robot::State robot_state, const std::vector<Npc> &obstacles)
{
    float closest_dist = std::numeric_limits<float>::max();
    for (const auto &obs : obstacles)
    {
        // Find the distance between the outer edges of circular robots
        auto const dist_edge_to_edge = (obs.position - robot_state.position).Length() - Npc::kRadius - Robot::kRadius;
        if (dist_edge_to_edge < closest_dist)
        {
            closest_dist = dist_edge_to_edge;
        }
    }

    return closest_dist;
}

bool isInsideBoundaries(const Robot::State &state)
{
    if ((state.position.x > 0) && (state.position.x < Robot::kAreaBoundaryX) &&
        (state.position.y < Robot::kAreaBoundaryY) && (state.position.y > 0))
    {
        return true;
    }
    return false;
}
} // namespace

Robot::Robot(const raylib::Vector2 pos_init, const float heading_init)
    : state_({pos_init, heading_init}), v_wheels_({0, 0})
{
}

Robot::State Robot::iterateKinematics(const Action &v_wheels, const State &state, const float dt) const
{
    State new_state;
    // A) straight line motion
    if (std::fabs(v_wheels.v_left - v_wheels.v_right) < 0.001F)
    {
        new_state.position.x = state.position.x + v_wheels.v_left * dt * cos(state.heading);
        new_state.position.y = state.position.y + v_wheels.v_left * dt * sin(state.heading);
        new_state.heading    = state.heading;
    }
    // B) Pure rotation
    else if (std::fabs(v_wheels.v_left + v_wheels.v_right) < 0.001F)
    {
        new_state.position.x = state.position.x;
        new_state.position.y = state.position.y;
        new_state.heading    = state.heading + (v_wheels.v_right - v_wheels.v_left) * dt / kWidth;
    }
    // C) Other
    else
    {
        const auto turn_radius_var =
            kRadius * (v_wheels.v_right + v_wheels.v_left) / (v_wheels.v_right - v_wheels.v_left);
        const auto delta_heading = (v_wheels.v_right - v_wheels.v_left) * dt / kWidth;

        new_state.position.x =
            state.position.x + turn_radius_var * (sin(delta_heading + state.heading) - sin(state.heading));
        new_state.position.y =
            state.position.y - turn_radius_var * (cos(delta_heading + state.heading) - cos(state.heading));
        new_state.heading = state.heading + delta_heading;
    }

    return new_state;
}

Robot::Action Robot::chooseAction(const std::vector<Npc> &obstacles_future,
                                  const Npc              &goal_future,
                                  const float             t_horizon,
                                  std::vector<State>     &possible_states_out)
{
    constexpr float kGoalWeight{20.F};              // weight associated to gaining distance on the target (benefit)
    constexpr float kObstacleWeight{6000.F};        // weight associated to proximity to the closest obstacle (cost)
    constexpr float kSafeDistToObs{Robot::kRadius}; // safe distance to the closest obstacle after which we penalize

    float         best_benefit{std::numeric_limits<float>::lowest()};
    Robot::Action chosen_action;
    possible_states_out.clear();

    std::array<float, 3> v_left_possibilities{v_wheels_.v_left - kMaxAcceleration * t_horizon,
                                              v_wheels_.v_left,
                                              v_wheels_.v_left + kMaxAcceleration * t_horizon};
    std::array<float, 3> v_right_possibilities{v_wheels_.v_right - kMaxAcceleration * t_horizon,
                                               v_wheels_.v_right,
                                               v_wheels_.v_right + kMaxAcceleration * t_horizon};

    for (const auto &v_l_possible : v_left_possibilities)
    {
        for (const auto &v_r_possible : v_right_possibilities)
        {
            std::cout << "vl: " << v_l_possible << " vr: " << v_r_possible << std::endl;
            bool max_vel_reached{(std::fabs(v_l_possible) >= kMaxVelocity) ||
                                 (std::fabs(v_r_possible) >= kMaxVelocity)};
            bool reversing{v_l_possible <= 0 && v_r_possible <= 0};
            if (!max_vel_reached && !reversing)
            {
                auto next_possible_state = iterateKinematics({v_l_possible, v_r_possible}, state_, t_horizon);
                if (!isInsideBoundaries(next_possible_state))
                {
                    continue;
                }
                else
                {
                    possible_states_out.push_back(next_possible_state);
                }
                // A) Negative cost related to obstacles
                auto closest_obs_dist = closestDistToObstacle(next_possible_state, obstacles_future);
                auto obstacle_penalty = 0.F;
                // Increase the collision cost linearly as we get closer within safe distance
                if (closest_obs_dist < kSafeDistToObs)
                {
                    obstacle_penalty = kObstacleWeight * (kSafeDistToObs - closest_obs_dist);
                }

                // B) Positive benefit related to goal
                auto current_goal_distance = (state_.position - goal_future.position).Length();
                auto future_goal_distance  = (next_possible_state.position - goal_future.position).Length();
                auto goal_distance_to_gain = current_goal_distance - future_goal_distance;
                auto goal_benefit          = kGoalWeight * goal_distance_to_gain;

                auto net_benefit = goal_benefit - obstacle_penalty;
                if (net_benefit > best_benefit)
                {
                    best_benefit          = net_benefit;
                    chosen_action.v_left  = v_l_possible;
                    chosen_action.v_right = v_r_possible;
                }
            }
        }
    }
    return chosen_action;
}

void Robot::draw() const
{
    raylib::Vector2 front = state_.position + raylib::Vector2(cos(state_.heading), sin(state_.heading)) * kRadius;
    DrawCircleLines(state_.position.x, state_.position.y, kRadius, BLUE);
    DrawLineEx(state_.position, front, 2.F, WHITE);

    // Calculate wheel positions
    raylib::Vector2 left_wheel_pos =
        state_.position + raylib::Vector2(cos(state_.heading + PI / 2), sin(state_.heading + PI / 2)) * kRadius;
    raylib::Vector2 right_wheel_pos =
        state_.position + raylib::Vector2(cos(state_.heading - PI / 2), sin(state_.heading - PI / 2)) * kRadius;

    // Draw wheels
    DrawCircleV(left_wheel_pos, kWheelRadius, YELLOW);
    DrawCircleV(right_wheel_pos, kWheelRadius, YELLOW);
}
