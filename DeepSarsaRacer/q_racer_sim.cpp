#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment.hpp"
#include "QAgent.hpp"

constexpr uint32_t kNumEpisodes{1500};
constexpr int16_t  kNumAgents{1};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    rl::Environment env(argv[1]);

    size_t          s_idx{700};
    rl::QLearnAgent q_agent(rl::QLearnAgent(
        {env.race_track_->track_data_points_.x_m[s_idx], env.race_track_->track_data_points_.y_m[s_idx]},
        env.race_track_->headings_[s_idx],
        0));
    env.setAgent(&q_agent);

    uint32_t           iteration{0};
    uint32_t           episode_idx{0};
    float              current_cum_reward{0.F};
    std::vector<float> all_cum_rewards;

    // for (uint32_t episode_idx{0}; episode_idx < kNumEpisodes; episode_idx++)
    while (true)
    {
        // --- Reset --- //
        iteration          = 0;
        current_cum_reward = 0.F;
        bool all_done{false};

        const auto start_idx = GetRandomValue(0, env.race_track_->track_data_points_.x_m.size() - 1);

        // q_agent.reset(
        //     {env.race_track_->track_data_points_.x_m[start_idx], env.race_track_->track_data_points_.y_m[start_idx]},
        //     env.race_track_->headings_[start_idx]);
        q_agent.reset({env.race_track_->track_data_points_.x_m[s_idx], env.race_track_->track_data_points_.y_m[s_idx]},
                      env.race_track_->headings_[s_idx]);

        // need to get an initial observation for the intial action, after reset

        env.step();
        q_agent.current_state_idx_ = q_agent.discretizeState();

        while (!all_done)
        {
            q_agent.updateAction(); // using the current_state from the most recent env step, determines the action
            env.step();             // agent moves in the environment with current_action, produces next_state

            size_t next_state_idx = q_agent.discretizeState();
            float  reward         = q_agent.reward();
            current_cum_reward += reward;
            q_agent.learn(q_agent.current_state_idx_, q_agent.current_action_idx_, reward, next_state_idx);

            if (q_agent.crashed_)
            {
                all_done = true;
                if (q_agent.epsilon_ > 0.05)
                    q_agent.epsilon_ *= rl::QLearnAgent::kEpsilonDiscount;
            }

            iteration++;
            q_agent.current_state_idx_ = next_state_idx;
        }

        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        std::cout << "eps: " << q_agent.epsilon_ << std::endl;
        all_cum_rewards.push_back(current_cum_reward);
        for (const auto r : all_cum_rewards)
        {
            std::cout << r << " ";
        }
        std::cout << std::endl;
        episode_idx++;
    }

    return 0;
}