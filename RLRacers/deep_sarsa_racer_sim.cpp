#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepSarsaAgent.hpp"
#include "Environment.hpp"
#include "GreedyAgent.hpp"

constexpr uint32_t kNumEpisodes{100};
constexpr int16_t  kNumAgents{1};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    deep_sarsa::Environment env(argv[1]);

    std::vector<deep_sarsa::DeepSarsaAgent> sarsa_agents;
    sarsa_agents.reserve(kNumAgents);
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        sarsa_agents.push_back(
            deep_sarsa::DeepSarsaAgent({env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx],
                                        env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]},
                                       env.race_track_->headings_[RaceTrack::kStartingIdx],
                                       i));
        env.setAgent(&sarsa_agents[i]);
    }

    auto greedy_agent(deep_sarsa::GreedyAgent({env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx],
                                               env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]},
                                              env.race_track_->headings_[RaceTrack::kStartingIdx],
                                              kNumAgents));
    greedy_agent.color_ = raylib::Color::SkyBlue();
    env.setAgent(&greedy_agent);

    uint32_t iteration{0};
    uint32_t episode_idx{0};

    // for (uint32_t episode_idx{0}; episode_idx < kNumEpisodes; episode_idx++)
    while (true)
    {
        // --- Reset --- //
        iteration = 0;
        bool all_done{false};

        const auto start_idx =
            torch::randint(0, env.race_track_->track_data_points_.x_m.size() - 1, {1}).item<int32_t>();
        for (auto &sarsa_agent : sarsa_agents)
        {
            sarsa_agent.reset({env.race_track_->track_data_points_.x_m[start_idx],
                               env.race_track_->track_data_points_.y_m[start_idx]},
                              env.race_track_->headings_[start_idx]);
        }
        greedy_agent.reset(
            {env.race_track_->track_data_points_.x_m[start_idx], env.race_track_->track_data_points_.y_m[start_idx]},
            env.race_track_->headings_[start_idx]);

        while (!all_done)
        {
            env.step(); // agent moves in the environment with current_action, produces next_state
            for (auto &sarsa_agent : sarsa_agents)
            {
                if (!sarsa_agent.crashed_)
                {
                    sarsa_agent.updateAction(); // using next_state from above, determines the next_action
                    sarsa_agent.train(
                        sarsa_agent.reward(),
                        sarsa_agent.isDone()); // uses current_state, current_action, next_state, next_action

                    sarsa_agent.state_tensor_       = sarsa_agent.next_state_tensor_;
                    sarsa_agent.current_action_     = sarsa_agent.next_action_;
                    sarsa_agent.current_action_idx_ = sarsa_agent.next_action_idx_;
                }
            }
            if (!greedy_agent.crashed_)
            {
                greedy_agent.updateAction();
            }

            iteration++;

            all_done = true;
            for (auto const &agent : sarsa_agents)
            {
                if (!agent.isDone())
                {
                    all_done = false;
                    break;
                }
            }
            if (!greedy_agent.isDone())
            {
                all_done = false;
            }
        }

        // Transfer the network from sarsa agent to greedy agent
        greedy_agent.setNetwork(deep_sarsa::DeepSarsaAgent::nn_);

        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        episode_idx++;
    }

    return 0;
}