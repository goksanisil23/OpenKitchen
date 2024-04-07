#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"
#include "Network.hpp"

namespace deep_sarsa
{
class GreedyAgent : public Agent
{
  public:
    static constexpr float kAccelerationDelta{1.0};
    static constexpr float kSteeringDeltaLow{0.2}; // degrees
    static constexpr float kSteeringDeltaHigh{0.8};

    const std::unordered_map<int32_t, std::pair<float, float>> kActionMap{
        {0, {kAccelerationDelta, 0.F}},
        {1, {-kAccelerationDelta, 0.F}},
        {2, {kAccelerationDelta, kSteeringDeltaLow}},
        {3, {kAccelerationDelta, kSteeringDeltaHigh}},
        {4, {kAccelerationDelta, -kSteeringDeltaLow}},
        {5, {kAccelerationDelta, -kSteeringDeltaHigh}},
        {6, {-kAccelerationDelta, kSteeringDeltaLow}},
        {7, {-kAccelerationDelta, kSteeringDeltaHigh}},
        {8, {-kAccelerationDelta, -kSteeringDeltaLow}},
        {9, {-kAccelerationDelta, -kSteeringDeltaHigh}}};

    GreedyAgent() = default;

    // Used when all agents are created initially, with randomized weights
    GreedyAgent(raylib::Vector2 start_pos, float start_rot, int16_t id)
        : Agent(start_pos, start_rot, id), nn_{Network()}
    {
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        // populate the input, but normalize the features so that all lie in [0,1]
        auto const         speed_norm = speed_ / kSpeedLimit; // since we don't allow (-) speed
        auto const         rot_norm   = normalizeAngleDeg(rot_) / kRotationLimit;
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        return torch::cat({torch::tensor({speed_norm}), torch::tensor({rot_norm}), torch::tensor(sensor_hits_norm)});
    }

    void setNetwork(const Network &source)
    {
        this->nn_.fc1->weight.data().copy_(source.fc1->weight.data());
        this->nn_.fc1->bias.data().copy_(source.fc1->bias.data());

        this->nn_.fc2->weight.data().copy_(source.fc2->weight.data());
        this->nn_.fc2->bias.data().copy_(source.fc2->bias.data());

        this->nn_.fc3->weight.data().copy_(source.fc3->weight.data());
        this->nn_.fc3->bias.data().copy_(source.fc3->bias.data());
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);
        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

    // Choose the action corresponding to maximum q-value estimate
    void updateAction()
    {
        auto       state_tensor = stateToTensor();
        auto       q_values     = nn_.forward(state_tensor);
        auto       action_idx   = q_values.argmax().item<int64_t>();
        auto const accel_steer_pair{kActionMap.at(action_idx)};
        current_action_.throttle_delta = accel_steer_pair.first;
        current_action_.steering_delta = accel_steer_pair.second;
    }

  private:
    Network nn_;
};

} // namespace deep_sarsa