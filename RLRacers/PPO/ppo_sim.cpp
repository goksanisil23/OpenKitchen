#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Environment.h"
#include "PPOAgent.hpp"

constexpr int16_t kNumAgents{15};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    std::vector<std::unique_ptr<rl::PPOAgent>> agents;
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.push_back(std::make_unique<rl::PPOAgent>(Vec2d{0, 0}, 0, i));
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

    bool all_done{false};
    while (true)
    {
        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        all_done = false;

        for (auto &agent : agents)
        {
            env.resetAgentAtRandomPoint(agent.get());
        }

        // need to get an initial observation for the intial action, after reset
        env.step();

        while (!all_done)
        {
            for (auto &agent : agents)
            {
                agent->current_state_tensor_ = agent->stateToTensor();
                agent->experience_buffer_.saved_states.push_back(agent->current_state_tensor_.detach());
                agent->updateAction();
                agent->experience_buffer_.saved_actions.push_back(agent->getCurrentAction());
            }

            env.step();

            all_done = true;
            for (auto &agent : agents)
            {
                agent->experience_buffer_.saved_rewards.push_back(1.0F);
                if (!agent->crashed_)
                {
                    // agent.experience_buffer_.saved_rewards.push_back(1.0F);
                    all_done = false;
                }
                // else
                // {
                //     agent.experience_buffer_.saved_rewards.push_back(-5.F);
                // }
            }
        }

        rl::PPOAgent::updatePolicy();

        episode_idx++;
    }

    return 0;
}