#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Environment.hpp"
#include "PotentialFieldAgent.hpp"
#include "VFHAgent.hpp"

using FieldAgent             = VFHAgent; // options = [PotFieldAgent,VFHAgent]
constexpr int16_t kNumAgents = 1;

int32_t pickResetPosition(const rl::Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

size_t getGoalPointIdx(const FieldAgent &agent, const rl::Environment &env)
{
    size_t current_idx = env.race_track_->findNearestTrackIndexBruteForce({agent.pos_.x, agent.pos_.y});
    size_t goal_index  = (current_idx + FieldAgent::kLookAheadIdx) % env.race_track_->track_data_points_.x_m.size();
    return goal_index;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    rl::Environment env(argv[1]);

    std::vector<FieldAgent> agents;
    agents.reserve(kNumAgents);
    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.emplace_back(FieldAgent({start_pos_x, start_pos_y},
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

            // need to get an initial observation for the intial action, after reset
            env.step();
        }

        while (!all_done)
        {
            for (auto &agent : agents)
            {
                size_t goal_index = getGoalPointIdx(agent, env);
                agent.setGoalPoint({env.race_track_->track_data_points_.x_m[goal_index],
                                    env.race_track_->track_data_points_.y_m[goal_index]});
                agent.updateAction();
            }

            env.step();

            all_done = true;
            for (auto &agent : agents)
            {
                if (!agent.crashed_)
                {
                    all_done = false;
                }
            }
        }

        episode_idx++;
        reset_idx = pickResetPosition(env, &agents.front());
    }

    return 0;
}