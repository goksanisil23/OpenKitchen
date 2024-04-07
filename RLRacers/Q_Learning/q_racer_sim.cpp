#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Environment.hpp"
#include "QAgent.hpp"

constexpr int16_t kNumAgents{30};
// create an agent which always chooses the greedy action based on latest q-table of other learning agents
constexpr bool kEnableGreedyAgent{false};
// at the end of each episode, take the average of all q-tables and distribute back to all agents
constexpr bool kShareCumulativeKnowledge{true};

int32_t pickResetPosition(const rl::Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

// Takes the average of state-action values among all agents and assigns it back to them.
void shareCumulativeKnowledge(std::vector<rl::QLearnAgent> &q_agents)
{
    // Take the average of action-value pairs and assign it to all agents
    std::array<std::array<float, rl::QLearnAgent::kActionSize>, rl::QLearnAgent::kNumStates> total_q_values;
    std::array<std::array<float, rl::QLearnAgent::kActionSize>, rl::QLearnAgent::kNumStates> valid_counts;
    for (auto &action_vals_per_state : total_q_values)
    {
        std::fill(action_vals_per_state.begin(), action_vals_per_state.end(), rl::QLearnAgent::kInvalidQVal);
    }
    for (auto &v : valid_counts)
    {
        std::fill(v.begin(), v.end(), 0.F);
    }

    for (size_t action_idx{0}; action_idx < rl::QLearnAgent::kActionSize; action_idx++)
    {
        for (size_t state_idx{0}; state_idx < rl::QLearnAgent::kNumStates; state_idx++)
        {
            float &total_val   = total_q_values.at(state_idx).at(action_idx);
            float &valid_count = valid_counts.at(state_idx).at(action_idx);
            for (const auto &q_agent : q_agents)
            {
                const auto &q_val{q_agent.q_table_.at(state_idx).at(action_idx)};
                if (q_val != rl::QLearnAgent::kInvalidQVal)
                {
                    if (total_val == rl::QLearnAgent::kInvalidQVal)
                    {
                        total_val = 0.F;
                    }
                    total_val += q_val;
                    valid_count += 1.F;
                }
            }
            if (valid_count > 0.F)
            {
                total_val /= valid_count;
            };
        }
    }

    // Assign to the agents
    for (size_t action_idx{0}; action_idx < rl::QLearnAgent::kActionSize; action_idx++)
    {
        for (size_t state_idx{0}; state_idx < rl::QLearnAgent::kNumStates; state_idx++)
        {
            for (auto &q_agent : q_agents)
            {
                q_agent.q_table_.at(state_idx).at(action_idx) = total_q_values.at(state_idx).at(action_idx);
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    rl::Environment env(argv[1]);

    std::vector<rl::QLearnAgent> q_agents;
    q_agents.reserve(kNumAgents);
    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        q_agents.emplace_back(
            rl::QLearnAgent({start_pos_x, start_pos_y},
                            env.race_track_->headings_[RaceTrack::kStartingIdx],
                            i,
                            env.race_track_->findNearestTrackIndexBruteForce({start_pos_x, start_pos_y}),
                            static_cast<int64_t>(env.race_track_->track_data_points_.x_m.size())));
        env.setAgent(&q_agents.back());
    }
    // (Optionally): Make one agent totally greedy to track the performance
    std::unique_ptr<rl::QLearnAgent> greedy_agent;
    if constexpr (kEnableGreedyAgent)
    {
        greedy_agent           = std::make_unique<rl::QLearnAgent>(raylib::Vector2{start_pos_x, start_pos_y},
                                                         env.race_track_->headings_[RaceTrack::kStartingIdx],
                                                         q_agents.size());
        greedy_agent->epsilon_ = 0.F;
        greedy_agent->color_   = BLUE;
        env.setAgent(greedy_agent.get());
    }

    uint32_t episode_idx{0};
    int32_t  reset_idx{RaceTrack::kStartingIdx};

    env.visualizer_->user_draw_callback_ = [&episode_idx, &q_agents]()
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
        if constexpr (kEnableGreedyAgent)
        {
            greedy_agent->reset({env.race_track_->track_data_points_.x_m[reset_idx],
                                 env.race_track_->track_data_points_.y_m[reset_idx]},
                                env.race_track_->headings_[reset_idx],
                                0);
        }

        // need to get an initial observation for the intial action, after reset
        env.step();
        for (auto &q_agent : q_agents)
        {
            q_agent.current_state_idx_ = q_agent.discretizeState();
        }
        if constexpr (kEnableGreedyAgent)
            greedy_agent->current_state_idx_ = greedy_agent->discretizeState();

        while (!all_done)
        {
            for (auto &q_agent : q_agents)
            {
                q_agent.updateAction(); // using the current_state from the most recent env step, determines the action
            }

            if constexpr (kEnableGreedyAgent)
                greedy_agent->updateAction();

            env.step(); // agent moves in the environment with current_action, produces next_state

            all_done = true;
            for (auto &q_agent : q_agents)
            {
                size_t next_state_idx = q_agent.discretizeState();
                float  reward =
                    q_agent.reward(env.race_track_->findNearestTrackIndexBruteForce({q_agent.pos_.x, q_agent.pos_.y}));
                q_agent.learn(q_agent.current_state_idx_, q_agent.current_action_idx_, reward, next_state_idx);
                if (!q_agent.crashed_)
                {
                    q_agent.current_state_idx_ = next_state_idx;
                    all_done                   = false;
                }
            }
            if constexpr (kEnableGreedyAgent)
            {
                if (!greedy_agent->crashed_)
                {
                    greedy_agent->current_state_idx_ = greedy_agent->discretizeState();
                    all_done                         = false;
                }
            }

            if (all_done)
            {
                for (auto &q_agent : q_agents)
                {
                    if (q_agent.epsilon_ > rl::QLearnAgent::kEpsilonDiscount)
                    {
                        q_agent.epsilon_ -= rl::QLearnAgent::kEpsilonDiscount;
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
        if constexpr (kShareCumulativeKnowledge)
            shareCumulativeKnowledge(q_agents);
        if constexpr (kEnableGreedyAgent)
            greedy_agent->q_table_ = q_agents.front().q_table_;
    }

    return 0;
}