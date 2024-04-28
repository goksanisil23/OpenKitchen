#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DDPGAgent.hpp"
#include "Environment/Environment.hpp"

constexpr int16_t kNumAgents{1};

int32_t pickResetPosition(const rl::Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    rl::Environment env(argv[1]);

    std::vector<rl::DDPGAgent> agents;
    agents.reserve(kNumAgents);
    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.emplace_back(rl::DDPGAgent({start_pos_x, start_pos_y},
                                          env.race_track_->headings_[RaceTrack::kStartingIdx],
                                          i,
                                          env.race_track_->findNearestTrackIndexBruteForce({start_pos_x, start_pos_y}),
                                          static_cast<int64_t>(env.race_track_->track_data_points_.x_m.size())));
        env.setAgent(&agents.back());
    }

    uint32_t episode_idx{0};
    int32_t  reset_idx{RaceTrack::kStartingIdx};

    env.visualizer_->user_draw_callback_ = [&episode_idx, &agents]()
    {
        // char buffer[30];
        // snprintf(buffer, sizeof(buffer), "Episode: %d eps: %.3f", episode_idx, agents.front().epsilon_);
        // DrawText(buffer, kScreenWidth - 250, 40, 20, YELLOW);
    };

    bool all_done{false};
    while (true)
    {
        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        all_done = false;

        for (auto &agent : agents)
        {
            agent.reset(
                {env.race_track_->track_data_points_.x_m[reset_idx],
                 env.race_track_->track_data_points_.y_m[reset_idx]},
                env.race_track_->headings_[reset_idx],
                env.race_track_->findNearestTrackIndexBruteForce({env.race_track_->track_data_points_.x_m[reset_idx],
                                                                  env.race_track_->track_data_points_.y_m[reset_idx]}));
        }

        // need to get an initial observation for the intial action, after reset
        env.step();

        for (auto &agent : agents)
        {
            agent.current_state_tensor_ = agent.stateToTensor();
        }

        while (!all_done)
        {
            for (auto &agent : agents)
            {
                agent.replay_buffer_.states.push(agent.getCurrentState());
                agent.updateAction();
                agent.replay_buffer_.actions.push(agent.current_action_);
            }

            env.step();

            all_done = true;
            for (auto &agent : agents)
            {
                agent.replay_buffer_.next_states.push(agent.getCurrentState());
                agent.replay_buffer_.rewards.push(1.0F);

                agent.current_state_tensor_ = agent.stateToTensor();

                if (!agent.crashed_)
                {
                    all_done = false;
                    agent.replay_buffer_.dones.push(0.0F);
                }
                else
                {
                    agent.replay_buffer_.dones.push(1.0F);
                }
            }
        }

        // Update the Actor-Critic networks with the samples from the replay buffer at the end of each episode
        for (auto &agent : agents)
        {
            agent.updateDDPG();
        }
        episode_idx++;
        reset_idx = pickResetPosition(env, &agents.front());
    }

    return 0;
}