#include <cassert>
#include <iostream>

#include "Agent.h"

#include "Typedefs.h"

Agent::Agent(Vec2d start_pos, float start_rot, int16_t id) : pos_{start_pos}, rot_{start_rot}, id_{id}
{
    // Setup the sensor pattern
    if (has_raycast_sensor_)
    { // Denser pattern towards the middle
        for (int i = -70; i <= 70; i += 1)
        {
            if (i % 10 == 0)
                sensor_ray_angles_.push_back(static_cast<float>(i));
        }
    }
}

void Agent::move()
{
    switch (movement_mode_)
    {
    case MovementMode::VELOCITY:
    {
        moveViaVelocity();
        break;
    }
    case MovementMode::ACCELERATION:
    {
        moveViaAcceleration();
        break;
    }
    case MovementMode::MANUAL:
    {
        moveViaUserInput();
        break;
    }
    default:
    {
        std::cerr << "Unimplemented movement mode." << std::endl;
        assert(false);
        break;
    }
    }
}

// Kinematics update of the driver, when controlled via keyboard by the user
void Agent::moveViaUserInput()
{
    // TODO
    // // constexpr float kAccInc         = 5.0f;
    // constexpr float kRotDeltaManual = 5.f;

    // if (IsKeyDown(KEY_RIGHT))
    // {
    //     rot_ += kRotDeltaManual;
    // }
    // else if (IsKeyDown(KEY_LEFT))
    // {
    //     rot_ -= kRotDeltaManual;
    // }

    // if (IsKeyDown(KEY_UP))
    // {
    //     // speed_ += kAccInc;

    //     pos_.x += cos(kDeg2Rad * rot_) * 10.0 * GetFrameTime();
    //     pos_.y += sin(kDeg2Rad * rot_) * 10.0 * GetFrameTime();
    // }
    // else if (IsKeyDown(KEY_DOWN))
    // {
    //     // speed_ -= kAccInc;

    //     pos_.x += cos(kDeg2Rad * rot_) * 10.0 * GetFrameTime();
    //     pos_.y += sin(kDeg2Rad * rot_) * 10.0 * GetFrameTime();
    // }
}

// Applies the most recent control actions to move the agent kinematically
void Agent::moveViaAcceleration()
{
    constexpr float kDt{0.016}; // ~60FPS, but set to constant to have determinism

    rot_ += current_action_.steering_delta;
    acceleration_ += current_action_.throttle_delta;
    speed_ += (acceleration_ * kDt);

    // speed_ = (speed_ < -kSpeedLimit) ? -kSpeedLimit : speed_;
    speed_ = (speed_ < 0) ? 0 : speed_;
    speed_ = (speed_ > kSpeedLimit) ? kSpeedLimit : speed_;

    float delta_x = cos(kDeg2Rad * rot_) * speed_ * kDt;
    pos_.x += delta_x;
    float delta_y = sin(kDeg2Rad * rot_) * speed_ * kDt;
    pos_.y += delta_y;
}

// Follows the center points of the track
void Agent::setPose(const Vec2d pos, const float rot)
{
    pos_ = pos;
    rot_ = rot;
}

// Movement based on direct setting of speed and steering
void Agent::moveViaVelocity()
{
    constexpr float kDt{0.016}; // ~60FPS, but set to constant to have determinism

    rot_ += current_action_.steering_delta;
    speed_ = current_action_.throttle_delta;

    float delta_x = cos(kDeg2Rad * rot_) * speed_ * kDt;
    pos_.x += delta_x;
    float delta_y = sin(kDeg2Rad * rot_) * speed_ * kDt;
    pos_.y += delta_y;
}

// Only resets the state in the environment and nothing related to the controller that might or might not need to
// propagate between episodes
void Agent::reset(const Vec2d &reset_pos, const float reset_rot)
{
    pos_          = reset_pos;
    rot_          = reset_rot;
    acceleration_ = 0.F;
    speed_        = 0.F;

    crashed_   = false;
    timed_out_ = false;
    completed_ = false;

    current_action_ = Action{0.F, 0.F};
}

// Returns whether the episode is done or not
bool Agent::isDone() const
{
    if (crashed_ || completed_)
        return true;
    else
        return false;
}