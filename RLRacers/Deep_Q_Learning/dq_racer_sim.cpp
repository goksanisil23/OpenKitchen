#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DQAgent.hpp"
#include "Environment.hpp"

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

    std::vector<rl::DQLearnAgent> q_agents;
    q_agents.reserve(kNumAgents);
    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        q_agents.emplace_back(
            rl::DQLearnAgent({start_pos_x, start_pos_y},
                             env.race_track_->headings_[RaceTrack::kStartingIdx],
                             i,
                             env.race_track_->findNearestTrackIndexBruteForce({start_pos_x, start_pos_y}),
                             static_cast<int64_t>(env.race_track_->track_data_points_.x_m.size())));
        env.setAgent(&q_agents.back());
    }

    uint32_t episode_idx{0};
    int32_t  reset_idx{RaceTrack::kStartingIdx};

    env.user_draw_callback_ = [&episode_idx, &q_agents]()
    {
        char buffer[30];
        snprintf(buffer, sizeof(buffer), "Episode: %d eps: %.3f", episode_idx, q_agents.front().epsilon_);
        DrawText(buffer, kScreenWidth - 250, 40, 20, YELLOW);
    };

    bool all_done{false};
    while (true)
    {
        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        std::cout << "eps: " << q_agents.front().epsilon_ << std::endl;

        // --- Reset --- //
        all_done = false;

        for (auto &q_agent : q_agents)
        {
            q_agent.reset(
                {env.race_track_->track_data_points_.x_m[reset_idx],
                 env.race_track_->track_data_points_.y_m[reset_idx]},
                env.race_track_->headings_[reset_idx],
                env.race_track_->findNearestTrackIndexBruteForce({env.race_track_->track_data_points_.x_m[reset_idx],
                                                                  env.race_track_->track_data_points_.y_m[reset_idx]}));
        }

        // need to get an initial observation for the intial action, after reset
        env.step();
        for (auto &q_agent : q_agents)
        {
            q_agent.current_state_tensor_ = q_agent.stateToTensor();
        }

        while (!all_done)
        {
            for (auto &q_agent : q_agents)
            {
                q_agent.updateAction(); // using the current_state from the most recent env step, determines the action
            }

            env.step(); // agent moves in the environment with current_action, produces next_state

            all_done = true;
            for (auto &q_agent : q_agents)
            {
                auto  next_state_tensor = q_agent.stateToTensor();
                float reward =
                    q_agent.reward(env.race_track_->findNearestTrackIndexBruteForce({q_agent.pos_.x, q_agent.pos_.y}));
                q_agent.learn(q_agent.current_state_tensor_, q_agent.current_action_idx_, reward, next_state_tensor);
                if (!q_agent.crashed_)
                {
                    q_agent.current_state_tensor_ = next_state_tensor;
                    all_done                      = false;
                }
            }

            if (all_done)
            {
                for (auto &q_agent : q_agents)
                {
                    if (q_agent.epsilon_ > rl::DQLearnAgent::kEpsilonDiscount)
                    {
                        q_agent.epsilon_ -= rl::DQLearnAgent::kEpsilonDiscount;
                    }
                    else
                    {
                        // Entirely greedy after sufficient exploration
                        q_agent.epsilon_ = 0.F;
                    }
                }
            }
        }

        episode_idx++;
        reset_idx = pickResetPosition(env, &q_agents.front());
        // Share the Q-tables across agents at the end of the episode
        // if constexpr (kShareCumulativeKnowledge)
        //     shareCumulativeKnowledge(q_agents);
    }

    return 0;
}