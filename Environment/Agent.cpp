#include "Agent.h"
#include "Typedefs.h"

Agent::Agent(raylib::Vector2 start_pos, float start_rot, int16_t id) : pos_{start_pos}, rot_{start_rot}, id_{id}
{
    // Setup the sensor pattern
    if (has_raycast_sensor_)
    { // Denser pattern towards the middle
        for (int i = -70; i <= 70; i += 1)
        {
            // if (std::abs(i) < 15)
            //     sensor_ray_angles_.push_back(static_cast<float>(i));
            // else if (std::abs(i) > 15 && i % 3 == 0)
            //     sensor_ray_angles_.push_back(static_cast<float>(i));
            if (i % 10 == 0)
                sensor_ray_angles_.push_back(static_cast<float>(i));
        }
    }
}

// Kinematics update of the driver, when controlled via keyboard by the user
void Agent::manualMove()
{
    if (manual_control_enabled_)
    {
        if (IsKeyDown(KEY_RIGHT))
        {
            rot_ += kRotSpeed;
        }
        else if (IsKeyDown(KEY_LEFT))
        {
            rot_ -= kRotSpeed;
        }

        if (IsKeyDown(KEY_UP))
        {
            // speed_ += kAccInc;

            pos_.x += cos(DEG2RAD * rot_) * 0.6 * GetFrameTime();
            pos_.y += sin(DEG2RAD * rot_) * 0.6 * GetFrameTime();
        }
        else if (IsKeyDown(KEY_DOWN))
        {
            // speed_ -= kAccInc;

            pos_.x += cos(DEG2RAD * rot_) * 0.6 * GetFrameTime();
            pos_.y += sin(DEG2RAD * rot_) * 0.6 * GetFrameTime();
        }

        // pos_.x += cos(DEG2RAD * rot_) * speed_ * GetFrameTime();
        // pos_.y += sin(DEG2RAD * rot_) * speed_ * GetFrameTime();
    }
}

// Applies the most recent control actions to move the agent kinematically
void Agent::move()
{
    constexpr float kDt{0.016}; // ~60FPS, but set to constant to have determinism

    if (auto_control_enabled_)
    {
        rot_ += current_action_.steering_delta;
        acceleration_ += current_action_.acceleration_delta;
        speed_ += (acceleration_ * kDt);

        // speed_ = (speed_ < -kSpeedLimit) ? -kSpeedLimit : speed_;
        speed_ = (speed_ < 0) ? 0 : speed_;
        speed_ = (speed_ > kSpeedLimit) ? kSpeedLimit : speed_;

        float delta_x = cos(DEG2RAD * rot_) * speed_ * kDt;
        pos_.x += delta_x;
        float delta_y = sin(DEG2RAD * rot_) * speed_ * kDt;
        pos_.y += delta_y;

        if ((std::abs(delta_x) < kDeltaStandstillLimit) && (std::abs(delta_y) < kDeltaStandstillLimit))
        {
            standstill_ctr_++;
            if (standstill_ctr_ > kStandstillTimeout)
            {
                standstill_timed_out_ = true;
            }
        }
        else
        {
            standstill_ctr_ = 0;
        }
    }
}

// Only resets the state in the environment and nothing related to the controller that might or might not need to
// propagate between episodes
void Agent::reset(const raylib::Vector2 &reset_pos, const float reset_rot)
{
    pos_          = reset_pos;
    rot_          = reset_rot;
    acceleration_ = 0.F;
    speed_        = 0.F;

    crashed_              = false;
    completed_            = false;
    standstill_timed_out_ = false;
    standstill_ctr_       = 0;

    score_ = 0.F;

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