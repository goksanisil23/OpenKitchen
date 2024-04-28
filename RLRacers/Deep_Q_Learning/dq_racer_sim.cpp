#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DQAgent.hpp"
#include "Environment/Environment.hpp"

constexpr int16_t kNumAgents{30};
// at the end of each episode, take the average of all q-tables and distribute back to all agents
constexpr bool kShareCumulativeKnowledge{true};

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
                dq_agent.replay_buffer_.actions.push(dq_agent.current_action_idx_);
            }

            env.step(); // agent moves in the environment with current_action, produces next_state

            all_done = true;
            for (auto &dq_agent : dq_agents)
            {
                // auto  next_state_tensor = dq_agent.stateToTensor();
                dq_agent.replay_buffer_.next_states.push(dq_agent.getCurrentState());
                float reward = dq_agent.reward(
                    env.race_track_->findNearestTrackIndexBruteForce({dq_agent.pos_.x, dq_agent.pos_.y}));
                dq_agent.replay_buffer_.rewards.push(reward);
                // dq_agent.learn(dq_agent.current_state_tensor_, dq_agent.current_action_idx_, reward, next_state_tensor);
                if (!dq_agent.crashed_)
                {
                    // dq_agent.current_state_tensor_ = next_state_tensor;
                    all_done = false;
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
        // Update DQ networks at the end of each episode in batched-fashion using replay buffer
        for (auto &dq_agent : dq_agents)
        {
            dq_agent.updateDQN();
        }

        episode_idx++;
        reset_idx = pickResetPosition(env, &dq_agents.front());
        // Share the Q-tables across agents at the end of the episode
        // if constexpr (kShareCumulativeKnowledge)
        //     shareCumulativeKnowledge(dq_agents);
    }

    return 0;
}