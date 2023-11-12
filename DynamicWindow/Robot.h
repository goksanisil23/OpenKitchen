#pragma once
#include "raylib-cpp.hpp"
#include <iomanip>

#include "Npc.h"

class Robot
{
  public:
    static constexpr float kRadius{25.0f};
    static constexpr float kWidth{kRadius * 2.F};
    static constexpr float kWheelRadius{1.0F};

    static constexpr float kMaxAcceleration{300.};
    static constexpr float kMaxVelocity{400.0};

    static constexpr int kAreaBoundaryX = 1600;
    static constexpr int kAreaBoundaryY = 1200;

    static constexpr int kNumHorizonStep{10}; // how many dt steps into the future we plan

    struct State
    {
        raylib::Vector2 position; // meters
        float           heading;  // radians
    };

    struct Action
    {
        float v_left;  // left wheel
        float v_right; // right wheel
    };

    State  state_;
    Action v_wheels_;

    Robot(const raylib::Vector2 pos_init, const float heading_init);

    State iterateKinematics(const Action &v_wheels, const State &state, const float dt) const;

    void draw() const;

    Robot::Action chooseAction(const std::vector<Npc> &obstacles_future,
                               const Npc              &goal_future,
                               const float             t_horizon,
                               std::vector<State>     &possible_states_out);
};