#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "Network.hpp"
#include "raylib_msgs.h"

namespace evo_driver
{
std::default_random_engine rand_generator;
class EvoController
{
  public:
    static constexpr float kOutputActivationLim{0.5F};
    struct EvoControls
    {
        float acceleration_delta;
        float steering_delta;
    };

    // Used after mating, to generate a new agent from a hybrid network
    EvoController(Network &&nn) : score_{0.F}
    {
        nn_ = std::move(nn);
    }

    // Used when all agents are created initially, with randomized weights
    EvoController() = default;

    void updateAction(const float speed, const float rotation, const std::vector<okitch::Vec2d> &sensor_meas)
    {
        float acceleration_delta{0.F};
        float steering_delta{0.F};

        // Inference gives values in [0,1], we'll use the ones that are > 0.5
        nn_output_ = nn_.infer(speed, rotation, sensor_meas);

        acceleration_delta += (nn_output_[0] > kOutputActivationLim) ? 0.01F : 0.F;
        acceleration_delta += (nn_output_[1] > kOutputActivationLim) ? -0.01F : 0.F;

        steering_delta += (nn_output_[2] > kOutputActivationLim) ? 0.05F : 0.F;  // left soft
        steering_delta += (nn_output_[3] > kOutputActivationLim) ? 0.2F : 0.F;   // left hard
        steering_delta += (nn_output_[4] > kOutputActivationLim) ? -0.05F : 0.F; // right soft
        steering_delta += (nn_output_[5] > kOutputActivationLim) ? -0.2F : 0.F;  // right hard

        controls_.acceleration_delta = acceleration_delta;
        controls_.steering_delta     = steering_delta;
    }

    void setScore(const size_t iters_before_crash)
    {
        score_ += static_cast<float>(iters_before_crash);
    }

  public:
    EvoControls                             controls_;
    Network                                 nn_;
    float                                   score_{0.F};
    std::array<float, Network::kOutputSize> nn_output_;
};

Network mate2Agents(const Network &n1, const Network &n2)
{
    constexpr float kMutationProb = 0.05;
    Network         offspring_nn;

    std::mt19937                     rand_gen(std::random_device{}());
    std::uniform_real_distribution<> rand_dist(0.0, 1.0);

    // For each weight coefficient in the network, we'll either pick random mutation, or one of the coefficient from the parents
    for (int32_t i{0}; i < n1.weights_1_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_1_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else if (rand_dist(rand_gen) < 0.5F) // pick 1st parent
        {
            offspring_nn.weights_1_.data()[i] = n1.weights_1_.data()[i];
        }
        else
        {
            offspring_nn.weights_1_.data()[i] = n2.weights_1_.data()[i];
        }
    }
    for (int32_t i{0}; i < n1.weights_2_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_2_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else if (rand_dist(rand_gen) < 0.5F) // pick 1st parent
        {
            offspring_nn.weights_2_.data()[i] = n1.weights_2_.data()[i];
        }
        else
        {
            offspring_nn.weights_2_.data()[i] = n2.weights_2_.data()[i];
        }
    }

    return offspring_nn;
}

void chooseAndMateAgents(std::vector<evo_driver::EvoController> &agents)
{
    std::vector<float> agent_scores;
    std::for_each(agents.begin(),
                  agents.end(),
                  [&agent_scores](const evo_driver::EvoController agent) { agent_scores.push_back(agent.score_); });

    // Probability of each agent to into mating is proportional to it's score
    std::random_device           rand_device;
    std::mt19937                 rand_generator(rand_device());
    std::discrete_distribution<> dist(agent_scores.begin(), agent_scores.end());

    std::vector<evo_driver::EvoController> new_agents;
    while (new_agents.size() < agents.size())
    {
        int32_t first_agent = dist(rand_generator);
        // Prevent self-mutation
        int32_t second_agent{-1};
        while ((second_agent == -1) || (first_agent == second_agent))
        {
            second_agent = dist(rand_generator);
        }
        // Mate the chosen agents
        std::cout << "mating agents " << first_agent << " " << second_agent << std::endl;
        std::cout << "scores: " << agents.at(first_agent).score_ << " " << agents.at(second_agent).score_ << std::endl;
        new_agents.emplace_back(mate2Agents(agents.at(first_agent).nn_, agents.at(second_agent).nn_));
    }
    agents = new_agents;
}

} // namespace evo_driver