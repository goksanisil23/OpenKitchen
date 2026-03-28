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
#include <thread>
#include <vector>

#include "../PotentialFieldAgent.hpp"
#include "Environment/Environment.h"

// static constexpr std::string_view kTrackName = "SaoPaulo.csv";
static constexpr std::string_view kTrackName = "";

constexpr bool kHiddenWindow{true};

constexpr size_t kNumGoalsBeforeReset{50};
constexpr size_t kTrajLength{10};

constexpr bool kResetAgentsRandomly{true};
constexpr bool kRandomizeLaneOnReset{true};
constexpr bool kRandomizeHeadingOnReset{true};

class DataCollectorAgent : public PotFieldAgent
{
  public:
    enum class MeasurementMode
    {
        Laser2d      = 1,
        BirdseyeView = 2
    };

    static constexpr MeasurementMode kMeasurementMode{MeasurementMode::BirdseyeView};
    static constexpr size_t          kBirdseyeSavePeriod{1}; // 10
    static constexpr float           kSteeringAngleClampDeg{10.0};

    DataCollectorAgent(const Vec2d start_pos, float start_rot, int16_t id) : PotFieldAgent(start_pos, start_rot, id)
    {

        // If directory_ exists, remove its contents, if it doesnt exist, create it
        if (!std::filesystem::exists(directory_))
        {
            std::filesystem::create_directory(directory_);
        }

        if constexpr (kMeasurementMode == MeasurementMode::BirdseyeView)
        {
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

    static std::string buildFilePrefix(const std::string &directory,
                                       const std::string &track_name,
                                       const size_t       episode_id,
                                       const size_t       step_id)
    {
        std::ostringstream ss;
        ss << directory << "/"
           << "birdseye_" << track_name << "_ep" << std::setw(4) << std::setfill('0') << episode_id << "_step"
           << std::setw(4) << std::setfill('0') << step_id;
        return ss.str();
    }

    void saveMeasurement(const std::string &track_name, const size_t episode_id, const size_t step_id, Environment &env)
    {
        const std::string filename = buildFilePrefix(directory_, track_name, episode_id, step_id);
        if constexpr (kMeasurementMode == MeasurementMode::BirdseyeView)
        {
            if (ctr_ % kBirdseyeSavePeriod == 0)
            {
                std::ofstream out{filename + ".txt"};
                out << current_action_.throttle_delta << " " << current_action_.steering_delta;
                out.close();
                env.saveImage(filename + ".png");
            }
        }
        else if (kMeasurementMode == MeasurementMode::Laser2d)
        {
            std::ofstream out{filename + ".txt"};
            for (const auto &hit : sensor_hits_)
            {
                out << hit.x << " " << hit.y << std::endl;
            }
            out << current_action_.throttle_delta << " " << current_action_.steering_delta;
        }
        else
        {
            std::cerr << "Unsupported measurement mode" << std::endl;
            std::abort();
        }
        ctr_++;
    }

    void removeEpisodeFiles(const std::string &track_name, const size_t episode_id, const size_t num_steps)
    {
        for (size_t s = 0; s < num_steps; ++s)
        {
            const std::string base = buildFilePrefix(directory_, track_name, episode_id, s);
            std::filesystem::remove(base + ".txt");
            std::filesystem::remove(base + ".png");
        }
    }

  public:
    const std::string directory_{kTrackName.empty() ? "measurements_and_actions_random"
                                                    : (std::string(kTrackName.substr(0, kTrackName.find_last_of('.'))) +
                                                       std::string("_random"))};
    size_t            ctr_{0};
};

size_t getGoalPointIdx(const DataCollectorAgent &agent, const Environment &env)
{
    const size_t current_idx = env.race_track_->findNearestTrackIndexBruteForce({agent.pos_.x, agent.pos_.y});
    const size_t goal_index =
        std::min(current_idx + PotFieldAgent::kLookAheadIdx, env.race_track_->track_data_points_.x_m.size() - 1);
    return goal_index;
}

Vec2d determineGoalPoint(const DataCollectorAgent &agent, const Environment &env, const size_t goal_index)
{
    return {env.race_track_->track_data_points_.x_m[goal_index], env.race_track_->track_data_points_.y_m[goal_index]};
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
            if (kTrackName == "")
            {
                track_files.push_back(entry.path());
            }
            else if (entry.path().filename() == kTrackName)
            {
                track_files.push_back(entry.path());
            }
        }
    }

    std::vector<std::unique_ptr<DataCollectorAgent>> agents;
    for (const auto &track_file : track_files)
    {
        agents.clear();
        agents.push_back(std::make_unique<DataCollectorAgent>(Vec2d{0, 0}, 0, 0));
        constexpr bool draw_rays{DataCollectorAgent::kMeasurementMode == DataCollectorAgent::MeasurementMode::Laser2d};

        Environment       env(track_file, createBaseAgentPtrs(agents), draw_rays, kHiddenWindow);
        const std::string track_name{env.race_track_->track_name_};

        float start_pos_x = env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx];
        float start_pos_y = env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx];

        auto const &agent = agents[0];

        agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);

        env.visualizer_->setAgentToFollow(agent.get());
        env.EnableScreenGrabber();

        // Sub-portion of the full race track
        size_t const    track_len = env.race_track_->track_data_points_.x_m.size();
        constexpr float kCloseEnoughToGoal{1.F};

        size_t goal_ctr{0};
        size_t episode_id{0};
        while (goal_ctr < kNumGoalsBeforeReset)
        {
            env.resetAgent(agent.get(), kResetAgentsRandomly, kRandomizeLaneOnReset, kRandomizeHeadingOnReset);

            const size_t start_index = getGoalPointIdx(*agent, env);
            const size_t end_idx     = std::min(start_index + kTrajLength, track_len - 1);
            auto const   end_pt      = determineGoalPoint(*agent, env, end_idx);
            size_t       step_id{0};
            bool         reached_goal{true};

            float dist_to_goal = Vec2d{agent->pos_.x - end_pt.x, agent->pos_.y - end_pt.y}.length();

            env.step();
            while ((dist_to_goal > kCloseEnoughToGoal) && !agent->crashed_)
            {
                const size_t goal_index = getGoalPointIdx(*agent, env);
                agent->setGoalPoint(determineGoalPoint(*agent, env, goal_index));
                agent->updateAction();

                agent->saveMeasurement(track_name, episode_id, step_id, env);
                step_id++;
                const float prev_dist_to_goal = dist_to_goal;
                env.step();
                dist_to_goal = Vec2d{agent->pos_.x - end_pt.x, agent->pos_.y - end_pt.y}.length();

                if (agent->crashed_)
                {
                    std::cerr << "CRASHED AT " << env.race_track_->track_name_ << std::endl;
                    reached_goal = false;
                }

                if (dist_to_goal > prev_dist_to_goal)
                {
                    std::cout << "Overshot goal pt, breaking out to reset" << std::endl;
                    reached_goal = false;
                    break;
                }
            }

            if (reached_goal)
            {
                ++goal_ctr;
            }
            else
            {
                agent->removeEpisodeFiles(track_name, episode_id, step_id);
            }
            ++episode_id;
            std::cout << "---- " << env.race_track_->track_name_ << ": " << goal_ctr << " / " << kNumGoalsBeforeReset
                      << std::endl;
        }
    }

    return 0;
}
