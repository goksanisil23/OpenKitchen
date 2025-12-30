#pragma once

#include <memory>

#include "Typedefs.h"

class Agent
{
  public:
    static constexpr float kSensorRange{200.F};
    static constexpr float kSpeedLimit{100.F};
    static constexpr float kRotationLimit{360.F};

    struct Action
    {
        float throttle_delta{0.F}; // unit depends on the MovementMode
        float steering_delta{0.F}; // [deg]
    };

    enum class MovementMode
    {
        VELOCITY     = 0, // agent moves by directly setting the velocity
        ACCELERATION = 1, // agent moves by setting the acceleration that affects velocity
        MANUAL       = 2  // agent moves by the human keyboard input
    };

    Agent() = default;

    Agent(Vec2d start_pos, float start_rot, int16_t id);

    virtual ~Agent()
    {
    }

    virtual void reset(const Vec2d &reset_pos, const float reset_rot);

    void move();
    void moveViaVelocity();
    void moveViaAcceleration();
    void moveViaUserInput();
    void setPose(const Vec2d pos, const float rot);
    bool isDone() const;
    void setMovementMode(const MovementMode mode)
    {
        movement_mode_ = mode;
    }

    inline void setHeadingDrawing(const bool draw_heading)
    {
        draw_agent_heading_ = draw_heading;
    }

    virtual void updateAction() = 0;

  public:
    Vec2d   pos_{};
    float   speed_{0.F};
    float   acceleration_{0.F};
    float   rot_{0.F}; // degrees
    float   radius_{9.0F};
    float   sensor_offset_{0.0F}; // distance of the sensor from the robot center, along heading direction
    int16_t id_{};
    int     color_[4]{80, 80, 80, 255}; // RGBA, Dark Gray

    bool has_raycast_sensor_{true};
    bool manual_control_enabled_{true};
    bool draw_agent_heading_{true};

    std::vector<float> sensor_ray_angles_;
    float              sensor_range_{kSensorRange};

    bool crashed_{false};
    bool completed_{false};
    bool timed_out_{false};

    std::vector<Vec2d> sensor_hits_;
    std::vector<Pixel> pixels_until_hit_;

    Action current_action_{0.F, 0.F};

    MovementMode movement_mode_{MovementMode::VELOCITY};
};

template <typename TDerivedAgent>
inline std::vector<Agent *> createBaseAgentPtrs(const std::vector<std::unique_ptr<TDerivedAgent>> &derived_agents)
{
    std::vector<Agent *> base_agent_ptrs;
    base_agent_ptrs.reserve(derived_agents.size());
    for (const auto &agent : derived_agents)
    {
        base_agent_ptrs.push_back(agent.get());
    }
    return base_agent_ptrs;
}