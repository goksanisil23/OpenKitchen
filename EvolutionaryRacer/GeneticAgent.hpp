#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Network.hpp"

namespace genetic
{
std::default_random_engine rand_generator;
class GeneticAgent : public Agent
{
  public:
    static constexpr float kOutputActivationLim{0.5F};

    static constexpr float kAccelerationDelta{0.3};
    static constexpr float kSteeringDeltaLow{1.0}; // degrees
    static constexpr float kSteeringDeltaHigh{4.0};

    GeneticAgent() = default;

    // Used when all agents are created initially, with randomized weights
    GeneticAgent(raylib::Vector2 start_pos, float start_rot, int16_t id)
        : Agent(start_pos, start_rot, id), nn_{Network()}
    {
    }

    void updateAction() override
    {
        float throttle_delta{0.F};
        float steering_delta{0.F};

        // Inference gives values in [0,1], we'll use the ones that are > 0.5
        nn_output_ = nn_.infer(speed_, rot_, sensor_hits_);

        if constexpr (Network::kClassificationLayerType == Network::ClassificationLayer::Sigmoid)
        {
            throttle_delta += (nn_output_[0] > kOutputActivationLim) ? kAccelerationDelta : 0.F;
            throttle_delta += (nn_output_[1] > kOutputActivationLim) ? -kAccelerationDelta : 0.F;

            steering_delta += (nn_output_[2] > kOutputActivationLim) ? kSteeringDeltaLow : 0.F;   // left soft
            steering_delta += (nn_output_[3] > kOutputActivationLim) ? kSteeringDeltaHigh : 0.F;  // left hard
            steering_delta += (nn_output_[4] > kOutputActivationLim) ? -kSteeringDeltaLow : 0.F;  // right soft
            steering_delta += (nn_output_[5] > kOutputActivationLim) ? -kSteeringDeltaHigh : 0.F; // right hard
        }
        else if constexpr (Network::kClassificationLayerType == Network::ClassificationLayer::Softmax)
        {
            const size_t max_output =
                std::distance(nn_output_.begin(), std::max_element(nn_output_.begin(), nn_output_.end()));

            switch (max_output)
            {
            case 0:
            {
                throttle_delta = kAccelerationDelta;
                break;
            }
            case 1:
            {
                throttle_delta = -kAccelerationDelta;
                break;
            }
            case 2:
            {
                steering_delta = kSteeringDeltaLow;
                break;
            }
            case 3:
            {
                steering_delta = kSteeringDeltaHigh;
                break;
            }
            case 4:
            {
                steering_delta = -kSteeringDeltaLow;
                break;
            }
            case 5:
            {
                steering_delta = -kSteeringDeltaHigh;
                break;
            }
            default:
            {
                throttle_delta = 0.F;
                steering_delta = 0.F;
                break;
            }
            }
        }
        else
        {
            assert(false && "Classification layer type unsupported");
        }

        current_action_.throttle_delta = throttle_delta;
        current_action_.steering_delta = steering_delta;
    }

  public:
    Network                                 nn_;
    std::array<float, Network::kOutputSize> nn_output_;
    float                                   score_{0.F};
};

} // namespace genetic