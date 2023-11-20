#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment.hpp"
#include "QAgent.hpp"

constexpr uint32_t kNumEpisodes{1500};
constexpr int16_t  kNumAgents{1};

int32_t pickResetPosition(const rl::Environment &env, const Agent *agent)
{
    auto crash_idx = env.race_track_->findNearestTrackIndexBruteForce({agent->pos_.x, agent->pos_.y});
    return crash_idx - 5;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    rl::Environment env(argv[1]);

    rl::QLearnAgent q_agent(rl::QLearnAgent({env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx],
                                             env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]},
                                            env.race_track_->headings_[RaceTrack::kStartingIdx],
                                            0));
    env.setAgent(&q_agent);

    uint32_t iteration{0};
    uint32_t episode_idx{0};
    int32_t  reset_idx{RaceTrack::kStartingIdx};

    env.user_draw_callback_ = [&episode_idx, &q_agent]()
    {
        std::string message = "Episode: " + std::to_string(episode_idx) + " eps: " + std::to_string(q_agent.epsilon_);
        DrawText(message.c_str(), kScreenWidth - 250, 20, 20, YELLOW);
    };

    // for (uint32_t episode_idx{0}; episode_idx < kNumEpisodes; episode_idx++)
    while (true)
    {
        // --- Reset --- //
        iteration = 0;
        bool all_done{false};

        q_agent.reset(
            {env.race_track_->track_data_points_.x_m[reset_idx], env.race_track_->track_data_points_.y_m[reset_idx]},
            env.race_track_->headings_[reset_idx]);

        // need to get an initial observation for the intial action, after reset
        env.step();
        q_agent.current_state_idx_ = q_agent.discretizeState();

        while (!all_done)
        {
            q_agent.updateAction(); // using the current_state from the most recent env step, determines the action
            env.step();             // agent moves in the environment with current_action, produces next_state

            size_t next_state_idx = q_agent.discretizeState();
            float  reward         = q_agent.reward();
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
        episode_idx++;
        reset_idx = pickResetPosition(env, &q_agent);
    }

    return 0;
}