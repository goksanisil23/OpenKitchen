#pragma once

#include <memory>

#include "Typedefs.h"
#include "raylib-cpp.hpp"

class Agent
{
  public:
    static constexpr bool  kDrawSensorRays{true};
    static constexpr float kSensorRange{200.F};
    static constexpr float kSpeedLimit{100.F};
    static constexpr float kRotationLimit{360.F};
    // displacement threshold used to indicate an agent is standstill
    static constexpr float kDeltaStandstillLimit{0.001F};
    // # of consecutive standstill iterations after which we reset the episode
    static constexpr uint32_t kStandstillTimeout{200};

    struct Action
    {
        float acceleration_delta{0.F};
        float steering_delta{0.F};
    };

    Agent() = default;

    Agent(raylib::Vector2 start_pos, float start_rot, int16_t id);

    virtual ~Agent()
    {
    }

    virtual void reset(const raylib::Vector2 &reset_pos, const float reset_rot);

    void move();
    void move2();
    void manualMove();
    void setPose(const raylib::Vector2 pos, const float rot);
    bool isDone() const;

    virtual void updateAction() = 0;

  public:
    raylib::Vector2 pos_{};
    float           speed_{0.F};
    float           acceleration_{0.F};
    float           rot_{0.F};
    float           radius_{9.0F};
    float           sensor_offset_{0.0F}; // distance of the sensor from the robot center, along heading direction
    int16_t         id_{};
    raylib::Color   color_{raylib::Color::DarkGray()};

    bool has_raycast_sensor_{true};
    bool manual_control_enabled_{true};
    bool auto_control_enabled_{true};

    std::vector<float> sensor_ray_angles_;
    float              sensor_range_{kSensorRange};

    bool     crashed_{false};
    bool     completed_{false};
    bool     standstill_timed_out_{false};
    uint32_t standstill_ctr_{0};

    static constexpr float kAccInc         = 5.0f;
    static constexpr float kRotDeltaManual = 5.f;

    std::vector<Vec2d> sensor_hits_;
    std::vector<Pixel> pixels_until_hit_;

    float score_{0.F};

    Action current_action_{0.F, 0.F};
};