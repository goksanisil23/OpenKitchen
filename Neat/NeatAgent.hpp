#pragma once

#include <array>

#include "Environment/Agent.h"
#include "NeatGenomeOps.hpp"

namespace neat
{
class NeatAgent : public Agent
{
  public:
    NeatAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        movement_mode_ = MovementMode::ACCELERATION;
    }

    void bindGenome(const Genome *genome)
    {
        genome_ = genome;
    }

    void updateAction() override
    {
        require(genome_ != nullptr, "NEAT agent is missing a bound genome");

        const std::array<float, Config::kNumOutputs> outputs = evaluateGenome(*genome_, sensor_hits_);
        current_action_.throttle_delta = ((2.0F * outputs[0]) - 1.0F) * Config::kThrottleDeltaLimit;
        current_action_.steering_delta = ((2.0F * outputs[1]) - 1.0F) * Config::kSteeringDeltaLimitDeg;
    }

  private:
    const Genome *genome_{nullptr};
};
} // namespace neat
