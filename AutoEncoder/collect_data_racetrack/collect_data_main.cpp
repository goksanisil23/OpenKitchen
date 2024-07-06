/* Runs through all the race tracks available under the given folder and generates pointcloud data

Usage:
./collect_data_racetracks path_to_racetracks/racetrack-database/tracks/

Pointclouds will be saved under point_clouds folder
*/

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment.hpp"
#include "Environment/Agent.h"

class DataCollectorAgent : public Agent
{
  public:
    DataCollectorAgent(raylib::Vector2 start_pos, float start_rot, int16_t id) : Agent(start_pos, start_rot, id)
    {

        if (!std::filesystem::exists(directory_))
        {
            assert(std::filesystem::create_directory(directory_));
        }
    }
    void updateAction() override
    {
        return;
    }

    void saveMeasurement(const std::string &track_name)
    {
        static size_t ctr{0};
        std::string   filename = directory_ + "/" + track_name + "_" + std::to_string(ctr) + ".txt";
        std::ofstream out{filename};
        for (const auto &hit : sensor_hits_)
        {
            out << hit.x << " " << hit.y << std::endl;
        }
        out.close();
        ctr++;
    }

  private:
    const std::string directory_{"point_clouds"};
};

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
            track_files.push_back(entry.path());
        }
    }
    for (const auto &track_file : track_files)
    {
        Environment       env(track_file);
        const std::string track_name{env.race_track_->track_name_};

        const float        start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
        const float        start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
        DataCollectorAgent agent({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx], 0);
        env.setAgent(&agent);

        size_t current_idx{RaceTrack::kStartingIdx};

        size_t end_idx = env.race_track_->track_data_points_.x_m.size();
        while (current_idx < end_idx)
        {
            raylib::Vector2 pos = {env.race_track_->track_data_points_.x_m[current_idx],
                                   env.race_track_->track_data_points_.y_m[current_idx]};
            float           rot = env.race_track_->headings_[current_idx];
            env.step(pos, rot); // agent moves in the environment with current_action, produces next_state
            agent.saveMeasurement(track_name);
            current_idx++;
        }
    }

    return 0;
}