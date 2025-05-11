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
constexpr bool    kRandomizeEnvResetPt{false};

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
        done = false;

        env.resetAgent(agent, kRandomizeEnvResetPt);

        // need to get an initial observation for the intial action, after reset
        env.step();

        agent->current_state_tensor_ = agent->stateToTensor();
        agent->prev_pos_             = agent->pos_;
        while (!done)
        {
            agent->updateAction();

            env.step();

            agent->current_state_tensor_ = agent->stateToTensor();
            float reward                 = (agent->pos_ - agent->prev_pos_).norm();
            if (agent->crashed_)
            {
                done   = true;
                reward = -5.F;
            }

            agent->policy_.rewards_.push_back(reward);
        }

        if (episode_idx % 10 == 0)
        {
            std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
            // std::cout << "std: " << std::exp(agent->policy_.log_std[0].item<float>()) << " "
            //           << std::exp(agent->policy_.log_std[1].item<float>()) << std::endl;
            std::cout << "throttle std: " << agent->avg_throttle_std_ << " steering std: " << agent->avg_steering_std_
                      << std::endl;
            std::cout << "avg reward: "
                      << std::accumulate(agent->policy_.rewards_.begin(), agent->policy_.rewards_.end(), 0.F) /
                             agent->policy_.rewards_.size()
                      << std::endl;
        }

        agent->updatePolicy();

        episode_idx++;

        if (env.isEnterPressed())
        {
            std::cout << "--- Switching to deterministic policy ---" << std::endl;
            break;
        }
    }
    {
        env.resetAgent(agent, kRandomizeEnvResetPt);
        env.step();
        agent->current_state_tensor_ = agent->stateToTensor();
        while (true)
        {
            agent->applyDeterministicAction();
            env.step();
            agent->current_state_tensor_ = agent->stateToTensor();
            if (agent->crashed_)
            {
                break;
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    return 0;
}