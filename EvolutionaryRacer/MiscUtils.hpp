// Visualization related utility functions for the genetic learning environment
#pragma once

#include <vector>

#include "Environment/RaceTrack.h"
#include "GeneticAgent.hpp"

namespace genetic
{
namespace util
{
float getAvgColonyScore(const std::vector<GeneticAgent> &agents)
{
    // Calculate the current average
    float current_avg{0.F};
    for (const auto &agent : agents)
    {
        current_avg += agent.score_;
    }
    current_avg /= static_cast<float>(agents.size());

    return current_avg;
}

void saveBestAgentNetwork(const std::vector<genetic::GeneticAgent> &agents)
{
    static float top_score_all_time{0.F};  // individual top score of an agent across all generations
    static float prev_gen_best_score{0.F}; // keeping track of the previous generation's best score

    size_t top_scorer_agent_id;
    float  best_score_current{0.F};
    for (size_t i{0}; i < agents.size(); i++)
    {

        if (agents[i].score_ > best_score_current)
        {
            best_score_current  = agents[i].score_;
            top_scorer_agent_id = i;
        }
    }

    if (prev_gen_best_score > best_score_current)
    {
        std::string error_string{"!! Current gen. high score " + std::to_string(best_score_current) +
                                 " is less than previous gen. high score " + std::to_string(prev_gen_best_score)};
        throw std::runtime_error(error_string);
        // std::cerr << error_string << std::endl;
    }
    prev_gen_best_score = best_score_current;

    if (best_score_current > top_score_all_time)
    {
        // Save the weights
        genetic::writeMatrixToFile("agent_weights_1.txt", agents[top_scorer_agent_id].nn_.weights_1_);
        genetic::writeMatrixToFile("agent_weights_2.txt", agents[top_scorer_agent_id].nn_.weights_2_);

        top_score_all_time = best_score_current;
    }

    // Compare previous action set
}

void assignScores(std::vector<GeneticAgent> &agents, const RaceTrack &race_track)
{
    for (auto &agent : agents)
    {
        auto nearest_track_idx = race_track.findNearestTrackIndexBruteForce({agent.pos_.x, agent.pos_.y});
        agent.score_           = static_cast<float>(nearest_track_idx);
    }
}
} // namespace util
} // namespace genetic