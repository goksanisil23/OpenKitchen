#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>

#include "Environment/Agent.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"

// Vector Field Histogram based navigation agent
class VFHAgent : public Agent
{
  public:
    static constexpr float   kVelocity{60.0};
    static constexpr float   kSteeringDelta{5};         // degrees
    static constexpr int32_t kObstacleDistThreshold{1}; // number of hits in the sector

    VFHAgent() = default;

    // Used when all agents are created initially, with randomized weights
    VFHAgent(const raylib::Vector2 start_pos,
             const float           start_rot,
             const int16_t         id,
             const size_t          start_idx     = 0,
             const size_t          track_idx_len = 0)
        : Agent(start_pos, start_rot, id)
    {
        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        const float start_angle = -90.F;
        for (int i = 0; i < 19; i++)
        {
            sensor_ray_angles_.push_back(i * 10 + start_angle);
        }
        //     sensor_ray_angles_.push_back(-90.F);
        // sensor_ray_angles_.push_back(-60.F);
        // sensor_ray_angles_.push_back(-30.F);
        // sensor_ray_angles_.push_back(0.F);
        // sensor_ray_angles_.push_back(30.F);
        // sensor_ray_angles_.push_back(60.F);
        // sensor_ray_angles_.push_back(90.F);

        num_sectors_ = sensor_ray_angles_.size();
        polar_histogram_.resize(num_sectors_, 0);
        binary_histogram_.resize(num_sectors_, false);
        fov_          = std::fabs(sensor_ray_angles_.back() - sensor_ray_angles_.front());
        sector_width_ = fov_ / num_sectors_;
    }

    void setGoalPoint(const raylib::Vector2 goal)
    {
        goal_point_ = goal;
    }

    void updateHistograms()
    {
        std::fill(polar_histogram_.begin(), polar_histogram_.end(), 0);

        for (size_t i{0}; i < sensor_ray_angles_.size(); i++)
        {
            if (sensor_hits_[i].norm() < Agent::kSensorRange)
            {
                size_t sector = (i / static_cast<float>(sensor_ray_angles_.size())) * static_cast<float>(num_sectors_);
                polar_histogram_[sector] += 1;
            }
        }

        for (int32_t i{0}; i < num_sectors_; i++)
        {
            binary_histogram_[i] = polar_histogram_[i] > kObstacleDistThreshold;
        }
    }

    int32_t findBestSector(const float goal_angle)
    {
        int32_t goal_sector = static_cast<int32_t>((goal_angle - sensor_ray_angles_.front()) / fov_ * num_sectors_);
        std::cout << " goal_sector " << goal_sector << std::endl;

        // Find the closest "free" sector to the desired sector
        for (int32_t i{0}; i < num_sectors_; i++)
        {
            if (!binary_histogram_[(goal_sector + i) % num_sectors_])
                return (goal_sector + i) % num_sectors_;

            if (!binary_histogram_[(goal_sector - i + num_sectors_) % num_sectors_])
                return (goal_sector - i + num_sectors_) % num_sectors_;
        }

        std::cerr << "[WARN]: No Free sector found!" << std::endl;
        return goal_sector; // fallback to the goal sector if no free sector is found
    }

    void updateAction()
    {
        std::cout << "robot pos:" << pos_.x << " " << pos_.y << std::endl;
        std::cout << "goal pos:" << goal_point_.x << " " << goal_point_.y << std::endl;
        updateHistograms();

        float goal_direction_in_world_frame = std::atan2(goal_point_.y - pos_.y, goal_point_.x - pos_.x) / M_PI * 180.f;
        std::cout << "goal angle in world frame: " << goal_direction_in_world_frame << std::endl;
        float goal_angle_in_robot_frame = goal_direction_in_world_frame - this->rot_;
        goal_angle_in_robot_frame       = fmod(goal_angle_in_robot_frame, 360.F);
        if (goal_angle_in_robot_frame > 180.F)
        {
            goal_angle_in_robot_frame -= 360.F;
        }
        else if (goal_angle_in_robot_frame <= -180.F)
        {
            goal_angle_in_robot_frame += 360.F;
        }
        std::cout << "goal angle in robot frame: " << goal_angle_in_robot_frame << std::endl;
        int32_t best_sector = findBestSector(goal_angle_in_robot_frame);
        std::cout << "best sector: " << best_sector << std::endl;

        float best_sector_angle = sector_width_ * best_sector + sensor_ray_angles_.front();

        std::cout << "best sector angle: " << best_sector_angle << std::endl;

        current_action_.acceleration_delta = 15.0;
        current_action_.steering_delta     = best_sector_angle;
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot, const size_t track_reset_idx)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

  private:
    raylib::Vector2 goal_point_;

    int32_t num_sectors_;
    float   sector_width_{};
    float   fov_{};

    std::vector<int32_t> polar_histogram_;  // counts number of sensor hits falling in that zone
    std::vector<bool>    binary_histogram_; // True means occupied
};