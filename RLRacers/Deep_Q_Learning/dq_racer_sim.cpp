#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DQAgent.hpp"
#include "Environment/Environment.h"

constexpr int16_t kNumAgents{30};

int32_t pickResetPosition(const Environment &env, const Agent *agent)
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

    Environment env(argv[1]);

    std::vector<rl::DQLearnAgent> dq_agents;
    dq_agents.reserve(kNumAgents);
    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        dq_agents.emplace_back(
            rl::DQLearnAgent({start_pos_x, start_pos_y},
                             env.race_track_->headings_[RaceTrack::kStartingIdx],
                             i,
                             env.race_track_->findNearestTrackIndexBruteForce({start_pos_x, start_pos_y}),
                             static_cast<int64_t>(env.race_track_->track_data_points_.x_m.size())));
        env.setAgent(&dq_agents.back());
    }

    uint32_t episode_idx{0};
    int32_t  reset_idx{RaceTrack::kStartingIdx};

    env.visualizer_->user_draw_callback_ = [&episode_idx, &dq_agents]()
    {
        char buffer[30];
        snprintf(buffer, sizeof(buffer), "Episode: %d eps: %.3f", episode_idx, dq_agents.front().epsilon_);
        DrawText(buffer, kScreenWidth - 250, 40, 20, YELLOW);
    };

    bool all_done{false};

    while (true)
    {
        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        std::cout << "eps: " << dq_agents.front().epsilon_ << std::endl;

        // --- Reset --- //
        all_done = false;

        for (auto &dq_agent : dq_agents)
        {
            dq_agent.reset(
                {env.race_track_->track_data_points_.x_m[reset_idx],
                 env.race_track_->track_data_points_.y_m[reset_idx]},
                env.race_track_->headings_[reset_idx],
                env.race_track_->findNearestTrackIndexBruteForce({env.race_track_->track_data_points_.x_m[reset_idx],
                                                                  env.race_track_->track_data_points_.y_m[reset_idx]}));
        }

        // need to get an initial observation for the intial action, after reset
        env.step();
        for (auto &dq_agent : dq_agents)
        {
            dq_agent.current_state_tensor_ = dq_agent.stateToTensor();
        }

        while (!all_done)
        {
            for (auto &dq_agent : dq_agents)
            {
                dq_agent.replay_buffer_.states.push(dq_agent.getCurrentState());
                dq_agent.updateAction();
                dq_agent.replay_buffer_.actions.push(dq_agent.getCurrentAction());
            }

            env.step();

            all_done = true;
            for (auto &dq_agent : dq_agents)
            {
                dq_agent.replay_buffer_.next_states.push(dq_agent.getCurrentState());
                // Both reward strategies below work
                float reward = dq_agent.calculateReward(
                    env.race_track_->findNearestTrackIndexBruteForce({dq_agent.pos_.x, dq_agent.pos_.y}));
                dq_agent.replay_buffer_.rewards.push(reward);
                // dq_agent.replay_buffer_.rewards.push(1.F);

                dq_agent.current_state_tensor_ = dq_agent.stateToTensor();

                if (!dq_agent.crashed_)
                {
                    all_done = false;
                    dq_agent.replay_buffer_.dones.push(0.0F);
                }
                else
                {
                    dq_agent.replay_buffer_.dones.push(1.0F);
                }
            }

            if (all_done)
            {
                for (auto &dq_agent : dq_agents)
                {
                    if (dq_agent.epsilon_ > rl::DQLearnAgent::kEpsilonDiscount)
                    {
                        dq_agent.epsilon_ -= rl::DQLearnAgent::kEpsilonDiscount;
                    }
                    else
                    {
                        // Entirely greedy after sufficient exploration
                        dq_agent.epsilon_ = 0.F;
                    }
                }
            }
        }
        // Update shared DQ-network at the end of each episode in batched-fashion using replay buffer
        rl::DQLearnAgent::updateDQN();

        episode_idx++;
        reset_idx = pickResetPosition(env, &dq_agents.front());
    }

    return 0;
}