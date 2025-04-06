#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "Environment/Environment.h"

namespace
{

std::vector<std::string> getRaceTrackFiles(const std::string &folder_path)
{
    namespace fs = std::filesystem;

    std::vector<std::string> track_files;
    for (const auto &entry : fs::directory_iterator(folder_path))
    {
        if (entry.path().extension() == ".csv")
        {
            track_files.push_back(entry.path().string());
        }
    }
    return track_files;
}

} // namespace

class GCLAgent : public Agent
{
  public:
    GCLAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {

        // Setup sensor
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-90.F);
        sensor_ray_angles_.push_back(-60.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(60.F);
        sensor_ray_angles_.push_back(90.F);

        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;

        // normalized_rays_.reserve(sensor_ray_angles_.size() * 2);
    }

    void updateAction() override
    {

        current_action_.throttle_delta = 0.5;
        current_action_.steering_delta = 0.F;
    }

    void reset(const Vec2d &reset_pos, const float reset_rot, const size_t track_reset_idx)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }
};

int32_t pickResetPosition(const Environment &env)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

int main(int argc, char **argv)
{

    auto const race_tracks = getRaceTrackFiles("/home/s0001734/Downloads/racetrack-database/tracks");

    const size_t N              = 10000; // Samples per iteration from expert and policy
    const size_t num_iterations = 1000;

    std::vector<std::unique_ptr<GCLAgent>> agents;
    agents.push_back(std::make_unique<GCLAgent>(Vec2d{0, 0}, 0, 0));
    agents.push_back(std::make_unique<GCLAgent>(Vec2d{0, 0}, 0, 1));
    std::vector<Agent *> agent_ptrs;
    for (const auto &agent_ptr : agents)
    {
        agent_ptrs.push_back(agent_ptr.get());
    }

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        // 2) Collect N samples with the current policy from the environment (Rollout)
        const size_t race_track_id_to_use{iter % race_tracks.size()};
        Environment  env(race_tracks[race_track_id_to_use], agent_ptrs);
        const float  start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
        const float  start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
        agents[0]->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx], 0);
        agents[1]->reset({start_pos_x, start_pos_y}, 180.F - env.race_track_->headings_[RaceTrack::kStartingIdx], 0);
        int32_t reset_idx{RaceTrack::kStartingIdx};
        auto    t0    = std::chrono::high_resolution_clock::now();
        int     iters = 0;
        for (size_t i = 0; i < N; i++)
        {
            for (auto &agent : agents)
            {
                if (agent->crashed_)
                {
                    reset_idx = pickResetPosition(env);
                    agent->reset({env.race_track_->track_data_points_.x_m[reset_idx],
                                  env.race_track_->track_data_points_.y_m[reset_idx]},
                                 env.race_track_->headings_[reset_idx],
                                 env.race_track_->findNearestTrackIndexBruteForce(
                                     {env.race_track_->track_data_points_.x_m[reset_idx],
                                      env.race_track_->track_data_points_.y_m[reset_idx]}));
                }
            }

            env.step(); // agent moves in the environment with current_action, produces next_state
            for (auto &agent : agents)
            {
                agent->updateAction();
            }

            // Check if 1 seconds has passed
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            iters++;
            if (dt > 1000)
            {
                std::cout << "FPS: " << iters / (dt / 1000.0) << std::endl;
                t0    = t1;
                iters = 0;
            }
        }
    }
    return 0;
}