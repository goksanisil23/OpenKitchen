#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Environment.h"
#include "ReinforceAgent.hpp"

constexpr int16_t kNumAgents{1};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    std::vector<std::unique_ptr<rl::ReinforceAgent>> agents;
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.push_back(std::make_unique<rl::ReinforceAgent>(Vec2d{0, 0}, 0, i));
    }
    Environment env(argv[1], createBaseAgentPtrs(agents));

    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (auto &agent : agents)
    {
        agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);
    }

    uint32_t episode_idx{0};

    env.visualizer_->user_draw_callback_ = [&episode_idx, &agents]()
    {
        char buffer[30];
        snprintf(buffer, sizeof(buffer), "Episode: %d", episode_idx);
        DrawText(buffer, kScreenWidth - 150, 40, 20, YELLOW);
    };

    bool                done{false};
    rl::ReinforceAgent *agent = agents[0].get();
    while (true)
    {
        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        done = false;

        env.resetAgentAtRandomPoint(agent);

        // need to get an initial observation for the intial action, after reset
        env.step();

        agent->current_state_tensor_ = agent->stateToTensor();

        Vec2d prev_pos = agent->pos_;
        while (!done)
        {
            agent->updateAction();

            env.step();

            agent->current_state_tensor_ = agent->stateToTensor();
            float reward                 = (agent->pos_ - prev_pos).norm();
            if (agent->crashed_)
            {
                done   = true;
                reward = -5.F;
            }
            agent->policy_.rewards.push_back(reward);
        }

        agent->updatePolicy();

        episode_idx++;
    }

    return 0;
}