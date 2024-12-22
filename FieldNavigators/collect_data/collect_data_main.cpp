/* Runs through all the race tracks available under the given folder and generates: 

Measurement mode is selected via kMeasurementMode:
A) 2d pointcloud measurement from robot at that step + (velocity + delta_steering) action
B) Birdseye image view from robot at that step + (velocity + delta_steering) action

Usage:
./collect_data_racetracks path_to_racetracks/racetrack-database/tracks/

A specific track can be selected by specifying kTrackName

Control inputs are bounded to (-+kSteeringAngleClampDeg) degrees for steering, and [0, 100] for throttle

Pointclouds or images will be saved under measurements_and_actions folder

Format
A) Pointcloud
- First N rows are 2d laser measurements
- Last row is the action:  Robot speed, steering delta
B) Birdseye image
- birdseye_TRACK_N.png (birdseye view RGB image)
- birdseye_TRACK_N.txt (action in 1 line)

*/

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../PotentialFieldAgent.hpp"
#include "Environment/Environment.h"

static constexpr std::string_view kTrackName = "IMS.csv";

class DataCollectorAgent : public PotFieldAgent
{
  public:
    enum class MeasurementMode
    {
        Laser2d      = 1,
        BirdseyeView = 2
    };

    static constexpr MeasurementMode kMeasurementMode{MeasurementMode::Laser2d};
    static constexpr size_t          kBirdseyeSavePeriod{1};
    static constexpr float           kSteeringAngleClampDeg{2.0};

    DataCollectorAgent(raylib::Vector2 start_pos, float start_rot, int16_t id) : PotFieldAgent(start_pos, start_rot, id)
    {

        if (!std::filesystem::exists(directory_))
        {
            assert(std::filesystem::create_directory(directory_));
        }
        if constexpr (kMeasurementMode == MeasurementMode::BirdseyeView)
        {
            this->setSensorRayDrawing(false);
            this->setHeadingDrawing(true);
        }
    }

    void updateAction() override
    {
        PotFieldAgent::updateAction();
        // Saturate it in degrees
        current_action_.steering_delta =
            std::clamp(current_action_.steering_delta, -kSteeringAngleClampDeg, kSteeringAngleClampDeg);
    }

    void saveMeasurement(const std::string &track_name, Environment &env, const int left_right_middle)
    {
        std::string filename;
        if constexpr (kMeasurementMode == MeasurementMode::BirdseyeView)
        {
            if (ctr_ % kBirdseyeSavePeriod == 0)
            {
                filename = directory_ + "/" + "birdseye_" + track_name + "_" + std::to_string(ctr_) + "_" +
                           std::to_string(left_right_middle) + ".txt";
                std::ofstream out{filename};
                out << current_action_.throttle_delta << " " << current_action_.steering_delta;
                out.close();
                filename.replace(filename.size() - 4, 4, ".png");
                // env.render_buffer_->Export(filename);
                env.visualizer_->saveImage(filename);
            }
        }
        else if (kMeasurementMode == MeasurementMode::Laser2d)
        {
            filename = directory_ + "/" + "laser2d_" + track_name + "_" + std::to_string(ctr_) + "_" +
                       std::to_string(left_right_middle) + ".txt";
            std::ofstream out{filename};
            for (const auto &hit : sensor_hits_)
            {
                out << hit.x << " " << hit.y << std::endl;
            }
            out << " " << current_action_.throttle_delta << " " << current_action_.steering_delta;
        }
        else
        {
            std::cerr << "Unsupported measurement mode" << std::endl;
            std::abort();
        }
        ctr_++;
    }

  public:
    // const std::string directory_{"measurements_and_actions"};
    const std::string directory_{"measurements_and_actions_IMS"};
    size_t            ctr_{0};
};

size_t getGoalPointIdx(const DataCollectorAgent &agent, const Environment &env)
{
    size_t current_idx = env.race_track_->findNearestTrackIndexBruteForce({agent.pos_.x, agent.pos_.y});
    size_t goal_index =
        std::min(current_idx + PotFieldAgent::kLookAheadIdx, env.race_track_->track_data_points_.x_m.size() - 1);
    return goal_index;
}

raylib::Vector2 determineGoalPoint(const DataCollectorAgent &agent,
                                   const Environment        &env,
                                   const size_t              goal_index,
                                   const int                 left_right_middle)
{
    constexpr bool kDisturbance{true};
    if constexpr (kDisturbance)
    {
        // Pick somewhere between the goal point and the nearest track boundary (laterally)
        auto            nearest_boundary_l = env.race_track_->left_bound_inner_[goal_index];
        auto            nearest_boundary_r = env.race_track_->right_bound_inner_[goal_index];
        raylib::Vector2 goal_pt            = {env.race_track_->track_data_points_.x_m[goal_index],
                                              env.race_track_->track_data_points_.y_m[goal_index]};
        if (left_right_middle == 0) // left
        {
            return {(nearest_boundary_l.x + goal_pt.x) / 2.F, (nearest_boundary_l.y + goal_pt.y) / 2.F};
        }
        else if (left_right_middle == 1) // right
        {
            return {(goal_pt.x + nearest_boundary_r.x) / 2.F, (goal_pt.y + nearest_boundary_r.y) / 2.F};
        }
        else // middle
        {
            return goal_pt;
        }
    }
    else
    {
        return {env.race_track_->track_data_points_.x_m[goal_index],
                env.race_track_->track_data_points_.y_m[goal_index]};
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a folder path for race track database\n";
        return -1;
    }

    std::vector<std::filesystem::path> track_files;
    for (const auto &entry : std::filesystem::directory_iterator(std::filesystem::path(argv[1])))
    {
        if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".csv")
        {
            if (entry.path().filename() == kTrackName)
            {
                // Add 3 times since we want to collect data for left, right and middle
                track_files.push_back(entry.path());
                track_files.push_back(entry.path());
                track_files.push_back(entry.path());
            }
        }
    }

    int left_right_middle{0}; // to arrange the disturbance

    for (const auto &track_file : track_files)
    {
        Environment       env(track_file);
        const std::string track_name{env.race_track_->track_name_};

        float start_pos_x, start_pos_y;
        if (left_right_middle == 0) // left
        {

            start_pos_x = (env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx] +
                           env.race_track_->left_bound_inner_[RaceTrack::kStartingIdx].x) /
                          2.F;
            start_pos_y = (env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx] +
                           env.race_track_->left_bound_inner_[RaceTrack::kStartingIdx].y) /
                          2.F;
        }
        else if (left_right_middle == 1) // right
        {
            start_pos_x = (env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx] +
                           env.race_track_->right_bound_inner_[RaceTrack::kStartingIdx].x) /
                          2.F;
            start_pos_y = (env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx] +
                           env.race_track_->right_bound_inner_[RaceTrack::kStartingIdx].y) /
                          2.F;
        }
        else // middle
        {
            start_pos_x = env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx];
            start_pos_y = env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx];
        }

        DataCollectorAgent agent({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx], 0);
        env.setAgent(&agent);
        env.visualizer_->setAgentToFollow(&agent);
        env.visualizer_->camera_.zoom = 10.0f;

        size_t goal_index = 0;
        size_t end_idx    = env.race_track_->track_data_points_.x_m.size() - 1;
        env.step();
        while (goal_index < end_idx)
        {
            goal_index = getGoalPointIdx(agent, env);
            agent.setGoalPoint(determineGoalPoint(agent, env, goal_index, left_right_middle));
            agent.updateAction();
            agent.saveMeasurement(track_name, env, left_right_middle);
            env.step(); // agent moves in the environment with current_action, produces next_state
            if (agent.crashed_)
            {
                std::cerr << "CRASHED AT " << env.race_track_->track_name_ << std::endl;
                return -1;
            }
        }
        std::cout << "---- " << env.race_track_->track_name_ << " " << left_right_middle << " DONE ----" << std::endl;
        left_right_middle = (left_right_middle + 1) % 3;
    }

    return 0;
}