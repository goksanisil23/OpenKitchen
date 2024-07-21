#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Visualizer.h"
#include "MiscUtils.hpp"
#include "VisUtils.hpp"

#include "raylib-cpp.hpp"
#include "spmc_queue.h"

#include "Environment/Environment.h"
#include "GeneticAgent.hpp"
#include "Mating.hpp"

constexpr int16_t kNumAgents{50};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    Environment env(argv[1]);

    std::vector<genetic::GeneticAgent> agents;
    agents.reserve(kNumAgents);
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.push_back(genetic::GeneticAgent({env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx],
                                                env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]},
                                               env.race_track_->headings_[0],
                                               i));
        env.setAgent(&agents.back());
    }

    bool reset_episode = false;

    uint32_t           episode_idx{0};
    uint32_t           iteration{0};
    std::vector<float> colony_avg_scores;
    bool               all_done{false};
    while (true)
    {
        all_done = false;
        if (reset_episode)
        {
            std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
            iteration = 0;
            episode_idx++;

            // First, assign scores based on how far along the track the agents has come
            genetic::util::assignScores(agents, *env.race_track_);
            genetic::util::saveBestAgentNetwork(agents);
            colony_avg_scores.emplace_back(genetic::util::getAvgColonyScore(agents));
            genetic::util::showColonyScore(colony_avg_scores);

            // Mate the agents before resetting
            genetic::chooseAndMateAgents(agents);

            for (auto &agent : agents)
            {
                agent.reset({env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx],
                             env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]},
                            env.race_track_->headings_[0]);
            }
            reset_episode = false;
        }

        // need to get an initial observation for the intial action, after reset
        env.step();
        while (!all_done)
        {
            for (auto &agent : agents)
            {
                agent.updateAction();
            }

            env.step();

            all_done = true;
            for (auto &agent : agents)
            {
                if (!agent.crashed_)
                {
                    all_done = false;
                }
            }
            iteration++;
            // genetic::util::drawActionBar(agents, iteration); //TODO
        }
    }
    return 0;
}