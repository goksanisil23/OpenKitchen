#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
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
        float acceleration_delta{0.F};
        float steering_delta{0.F};
    };

    // Used after mating, to generate a new agent from a hybrid network
    EvoController(Network &&nn) : score_{0.F}
    {
        nn_ = std::move(nn);
    }

    void reset()
    {
        controls_ = EvoControls();
        score_    = 0;
        actions_history_.clear();
        meas_history_.clear();
    }

    // Used when all agents are created initially, with randomized weights
    EvoController() = default;

    void updateAction2(const float speed, const float rotation, const std::vector<okitch::Vec2d> &sensor_meas)
    {
        // for (const auto mes : sensor_meas)
        // {
        //     std::cout << mes.y << " ";
        // }
        // std::cout << std::endl;

        float sum_x{0.F};
        float sum_y{0.F};
        for (auto const meas : sensor_meas)
        {
            sum_x += meas.x;
        }
        for (auto const meas : sensor_meas)
        {
            sum_y += meas.y;
        }

        if (sum_x > 0.)
            controls_.acceleration_delta = 0.5;
        else
            controls_.acceleration_delta = 0.3;
        if (sum_y > 0.)
            controls_.steering_delta = .3;
        else
            controls_.steering_delta = -.3;
    }

    void updateAction(const float speed, const float rotation, const std::vector<okitch::Vec2d> &sensor_meas)
    {
        constexpr float kAccelerationDelta{0.3};
        constexpr float kSteeringDeltaLow{1.0}; // degrees
        constexpr float kSteeringDeltaHigh{4.0};
        float           acceleration_delta{0.F};
        float           steering_delta{0.F};

        // Inference gives values in [0,1], we'll use the ones that are > 0.5
        nn_output_ = nn_.infer(speed, rotation, sensor_meas);

        if constexpr (Network::kClassificationLayerType == Network::ClassificationLayer::Sigmoid)
        {
            acceleration_delta += (nn_output_[0] > kOutputActivationLim) ? kAccelerationDelta : 0.F;
            acceleration_delta += (nn_output_[1] > kOutputActivationLim) ? -kAccelerationDelta : 0.F;

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
                acceleration_delta = kAccelerationDelta;
                break;
            }
            case 1:
            {
                acceleration_delta = -kAccelerationDelta;
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
                acceleration_delta = 0.F;
                steering_delta     = 0.F;
                break;
            }
            }
        }
        else
        {
            assert(false && "Classification layer type unsupported");
        }

        controls_.acceleration_delta = acceleration_delta;
        controls_.steering_delta     = steering_delta;

        actions_history_.push_back(controls_);
        meas_history_.push_back(sensor_meas);
    }

  public:
    EvoControls                             controls_{0.F, 0.F};
    Network                                 nn_;
    float                                   score_{0.F};
    std::array<float, Network::kOutputSize> nn_output_;
    std::vector<EvoControls>                actions_history_;
    std::vector<std::vector<okitch::Vec2d>> meas_history_;
};

void checkRegression(const std::vector<evo_driver::EvoController> &agents,
                     const std::vector<evo_driver::EvoController> &new_agents)
{
    static float                                   prev_best_score{0};
    static std::vector<EvoController::EvoControls> prev_leader_controls;
    static std::vector<std::vector<okitch::Vec2d>> prev_leader_meas;
    static std::array<Eigen::MatrixXf, 2>          prev_leader_network;

    if (prev_best_score > new_agents.front().score_)
    {
        std::cout << "REGRESSED" << std::endl;

        // Actions of the previous episode's best performer
        for (size_t i{0}; i < agents.front().actions_history_.size(); i++)
        {
            if (agents.front().actions_history_[i].acceleration_delta != prev_leader_controls[i].acceleration_delta)
            {
                std::cout << "acc delta deviated at " << i << ": "
                          << agents.front().actions_history_[i].acceleration_delta << " -> "
                          << prev_leader_controls[i].acceleration_delta << std::endl;
                break;
            }
            if (agents.front().actions_history_[i].steering_delta != prev_leader_controls[i].steering_delta)
            {
                std::cout << "str delta deviated at " << i << ": " << agents.front().actions_history_[i].steering_delta
                          << " -> " << prev_leader_controls[i].steering_delta << std::endl;
                break;
            }
        }
        bool break_meas{false};
        for (size_t j{0}; j < agents.front().meas_history_.size(); j++)
        {
            for (size_t i{0}; i < agents.front().meas_history_[j].size(); i++)
            {
                if (agents.front().meas_history_[j][i].x != prev_leader_meas[j][i].x)
                {
                    std::cout << "meas(x) deviated at step" << j << " index " << i << " : "
                              << agents.front().meas_history_[j][i].x << " -> " << prev_leader_meas[j][i].x
                              << std::endl;
                    break_meas = true;
                    break;
                }
                if (agents.front().meas_history_[j][i].y != prev_leader_meas[j][i].y)
                {
                    std::cout << "meas(y) deviated at step" << j << " index " << i << " : "
                              << agents.front().meas_history_[j][i].y << " -> " << prev_leader_meas[j][i].y
                              << std::endl;
                    break_meas = true;
                    break;
                }
            }
            if (break_meas)
            {
                break;
            }
        }

        for (int k{0}; k < prev_leader_network[0].rows() * prev_leader_network[0].cols(); k++)
        {
            if (prev_leader_network[0].data()[k] != agents.front().nn_.weights_1_.data()[k])
            {
                printf("networks(1) differ at coef %d: %f %f\n",
                       k,
                       prev_leader_network[0].data()[k],
                       agents.front().nn_.weights_1_.data()[k]);
                // break;
            }
        }
        for (int k{0}; k < prev_leader_network[1].rows() * prev_leader_network[1].cols(); k++)
        {
            if (prev_leader_network[1].data()[k] != agents.front().nn_.weights_2_.data()[k])
            {
                printf("networks(2) differ at coef %d: %f %f\n",
                       k,
                       prev_leader_network[0].data()[k],
                       agents.front().nn_.weights_1_.data()[k]);
                // break;
            }
        }
    }
    prev_best_score      = agents.front().score_;
    prev_leader_controls = agents.front().actions_history_;
    prev_leader_meas     = agents.front().meas_history_;
    prev_leader_network  = {agents.front().nn_.weights_1_, agents.front().nn_.weights_2_};
}

// Mate 2 agents s.t. the offsprint node is direct transfer of one of the parent's node, whose probability depends on the episode score
Network mate2AgentsSelective(const evo_driver::EvoController &agent_1, const evo_driver::EvoController &agent_2)
{
    constexpr float kMutationProb        = 0.05; // 0.05 probability for a given weight to mutate randomly
    constexpr float kDominantAgentThresh = 0.75; // 0.75 favoribility of the dominant agent during mating
    Network         offspring_nn;

    const Network &n1 = (agent_1.score_ > agent_2.score_) ? agent_1.nn_ : agent_2.nn_; // dominant agent
    const Network &n2 = (agent_1.score_ > agent_2.score_) ? agent_2.nn_ : agent_1.nn_;

    std::mt19937                     rand_gen(std::random_device{}());
    std::uniform_real_distribution<> rand_dist(0.0, 1.0);

    // For each weight coefficient in the network, we'll either pick random mutation, or one of the coefficient from the parents
    for (int32_t i{0}; i < n1.weights_1_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_1_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else if (rand_dist(rand_gen) < kDominantAgentThresh) // pick 1st parent
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
        else if (rand_dist(rand_gen) < kDominantAgentThresh) // pick 1st parent
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

// Mates 2 agents s.t. the offspring node is the weighted average of the parents' nodes
Network mate2AgentsAvg(const evo_driver::EvoController &agent_1, const evo_driver::EvoController &agent_2)
{
    constexpr float kMutationProb        = 0.05; // 0.05 probability for a given weight to mutate randomly
    constexpr float kDominantAgentWeight = 0.75; // 0.75 favoribility of the dominant agent during mating
    constexpr float kPassiveAgentWeight  = 1.F - kDominantAgentWeight;
    Network         offspring_nn;

    const Network &n1 = (agent_1.score_ > agent_2.score_) ? agent_1.nn_ : agent_2.nn_; // dominant agent
    const Network &n2 = (agent_1.score_ > agent_2.score_) ? agent_2.nn_ : agent_1.nn_;

    std::mt19937                     rand_gen(std::random_device{}());
    std::uniform_real_distribution<> rand_dist(0.0, 1.0);

    // For each weight coefficient in the network, we'll either pick random mutation, or one of the coefficient from the parents
    for (int32_t i{0}; i < n1.weights_1_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_1_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else
        {
            offspring_nn.weights_1_.data()[i] =
                n1.weights_1_.data()[i] * kDominantAgentWeight + n2.weights_1_.data()[i] * kPassiveAgentWeight;
        }
    }
    for (int32_t i{0}; i < n1.weights_2_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_2_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else
        {
            offspring_nn.weights_2_.data()[i] =
                n1.weights_2_.data()[i] * kDominantAgentWeight + n2.weights_2_.data()[i] * kPassiveAgentWeight;
        }
    }

    return offspring_nn;
}

/* *** Mating strategy ***
- 1 clone of the fittest agent always goes into next generation
- 1 self-mutation of the fittest agent always goes into next generation
- Total size of the colony is kept constant
- Only top N-agents are chosen for mating
- Probability to be chosen for mating is proportional to the agent score
*/
void chooseAndMateAgents(std::vector<evo_driver::EvoController> &agents)
{
    const size_t                         colony_size{agents.size()}; // original size of the colony
    std::vector<float>                   agent_scores;
    std::unordered_map<int32_t, int32_t> agent_to_num_chosen_map; // which agent is chosen how many times

    // Filter so that only the top N scores can be parents
    std::sort(agents.begin(),
              agents.end(),
              [](const auto agent_1, const auto agent_2) { return agent_1.score_ > agent_2.score_; });
    constexpr size_t kNumParents{5};
    agents.resize(kNumParents);

    for (int32_t i{0}; i < static_cast<int32_t>(kNumParents); i++)
    {
        agent_scores.push_back(agents[i].score_);
        agent_to_num_chosen_map.insert({i, 0});
    }

    // Top agent clone and mutation are always chosen at least once
    std::vector<evo_driver::EvoController> new_agents;
    new_agents.push_back(agents.front());
    new_agents.push_back(mate2AgentsSelective(agents.front(), agents.front()));
    agent_to_num_chosen_map.at(0) += 2;
    std::cout << "top: " << new_agents.front().score_ << std::endl;

    checkRegression(agents, new_agents);

    // Probability of each agent to into mating is proportional to it's score
    std::random_device           rand_device;
    std::mt19937                 rand_generator(rand_device());
    std::discrete_distribution<> dist(agent_scores.begin(), agent_scores.end());

    while (new_agents.size() < colony_size)
    {
        int32_t first_agent = dist(rand_generator);
        agent_to_num_chosen_map.at(first_agent)++;
        // Prevent self-mutation
        int32_t second_agent{-1};
        while ((second_agent == -1) || (first_agent == second_agent))
        {
            second_agent = dist(rand_generator);
        }
        agent_to_num_chosen_map.at(second_agent)++;
        // Mate the chosen agents
        new_agents.emplace_back(mate2AgentsSelective(agents.at(first_agent), agents.at(second_agent)));
    }
    // Summarize who's chosen how many times
    for (size_t id{0}; id < agents.size(); id++)
    {
        std::cout << std::left << std::setw(15) << "agent score: " << std::right << std::setw(6) << agents[id].score_
                  << " # chosen: " << std::right << std::setw(6) << agent_to_num_chosen_map.at(id) << std::endl;
    }

    // Sanity check
    {
        for (int k{0}; k < agents[0].nn_.weights_1_.rows() * agents[0].nn_.weights_1_.cols(); k++)
        {
            if (agents[0].nn_.weights_1_.data()[k] != new_agents[0].nn_.weights_1_.data()[k])
            {
                printf("------- networks(1) differ at coef %d: %f %f\n",
                       k,
                       agents[0].nn_.weights_1_.data()[k],
                       new_agents[0].nn_.weights_1_.data()[k]);
                // break;
            }
        }
        for (int k{0}; k < agents[0].nn_.weights_2_.rows() * agents[0].nn_.weights_2_.cols(); k++)
        {
            if (agents[0].nn_.weights_2_.data()[k] != agents.front().nn_.weights_2_.data()[k])
            {
                printf("------- networks(2) differ at coef %d: %f %f\n",
                       k,
                       agents[0].nn_.weights_2_.data()[k],
                       new_agents[0].nn_.weights_2_.data()[k]);
                // break;
            }
        }
    }

    agents = new_agents;
}

void chooseAndMateAgents2(std::vector<evo_driver::EvoController> &agents)
{
    std::vector<evo_driver::EvoController> new_agents;

    for (size_t i{0}; i < agents.size(); i++)
    {
        new_agents.push_back(agents[i]);
    }

    agents = new_agents;
}

} // namespace evo_driver