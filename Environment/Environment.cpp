#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "Environment.h"
#include "raylib-cpp.hpp"

namespace
{
void checkAndUpdateStandstill(const Agent &agent, DisplacementStats &displacement_stats)
{
    if (displacement_stats.displacement_ctr == 0)
    {
        displacement_stats.init_pos               = agent.pos_;
        displacement_stats.displacement_timed_out = false;
        displacement_stats.displacement_ctr++;
        return;
    }
    if (displacement_stats.displacement_ctr >= DisplacementStats::kPeriod)
    {
        const float dist_moved = agent.pos_.distanceSquared(displacement_stats.init_pos);
        if (dist_moved < DisplacementStats::kDisplamentThreshold * DisplacementStats::kDisplamentThreshold)
        {
            displacement_stats.displacement_timed_out = true;
        }
        displacement_stats.displacement_ctr = 0;
    }
    else
    {
        displacement_stats.displacement_timed_out = false;
        displacement_stats.displacement_ctr++;
    }
}
} // namespace

Environment::Environment(const std::string          &race_track_path,
                         const std::vector<Agent *> &agents,
                         const bool                  draw_rays,
                         const bool                  hidden_window)
    : draw_rays_(draw_rays)
{
    race_track_     = std::make_unique<RaceTrack>(race_track_path);
    track_segments_ = std::make_unique<TrackSegments>(*race_track_);
    visualizer_     = std::make_unique<env::Visualizer>(hidden_window);

    // Initialize the collision checker with track segments (vector-based collision)
    collision_checker_ = std::make_unique<CollisionChecker>(
        track_segments_->getDeviceSegments(), track_segments_->getNumSegments(), agents);

    screen_grabber_ = std::make_unique<ScreenGrabber>(kScreenWidth, kScreenHeight);

    // Store pointers to the agents
    agents_ = agents;

    displacement_stats_.resize(agents.size());
}

void Environment::drawSensorRanges(const std::vector<Vec2d> &sensor_hits)
{
    std::string range_string{""};
    for (const auto &hit : sensor_hits)
    {
        range_string += std::to_string(static_cast<int64_t>(hit.norm())) + " ";
    }
    DrawText(range_string.c_str(), kScreenWidth - 400, 100, 15, WHITE);
}

int32_t Environment::pickRandomResetTrackIdx() const
{
    return GetRandomValue(0, static_cast<int32_t>(race_track_->track_data_points_.x_m.size()) - 1);
}

void Environment::resetAgent(Agent     *agent,
                             const bool pick_random_point,
                             const bool randomize_lane,
                             const bool randomize_heading)
{
    int32_t reset_idx = pick_random_point ? pickRandomResetTrackIdx() : RaceTrack::kStartingIdx;

    // If randomizing heading, add a random offset in [-45, 45] degrees
    float         heading_offset = 0.F;
    static size_t ctr            = 0;
    if (pick_random_point && randomize_heading)
    {
        constexpr float kHeadingRandomizationRangeDeg{45.F};
        heading_offset = static_cast<float>(GetRandomValue(0, kHeadingRandomizationRangeDeg));
        if (ctr % 2 == 0)
        {
            heading_offset = (heading_offset + kHeadingRandomizationRangeDeg) * -1.F;
        }
        else
        {
            heading_offset = heading_offset + kHeadingRandomizationRangeDeg;
        }
        ctr++;
    }

    float start_pos_x;
    float start_pos_y;
    // Find a random point between the left and right lane boundaries. Alpha in [0, 1]
    if (pick_random_point && randomize_lane)
    {
        auto const  nearest_lane_boundary_l = race_track_->left_bound_inner_[reset_idx];
        auto const  nearest_lane_boundary_r = race_track_->right_bound_inner_[reset_idx];
        const float alpha                   = static_cast<float>(GetRandomValue(10, 90)) / 100.F;
        start_pos_x = nearest_lane_boundary_l.x * alpha + nearest_lane_boundary_r.x * (1.F - alpha);
        start_pos_y = nearest_lane_boundary_l.y * alpha + nearest_lane_boundary_r.y * (1.F - alpha);
    }
    else
    {
        start_pos_x = race_track_->track_data_points_.x_m[reset_idx];
        start_pos_y = race_track_->track_data_points_.y_m[reset_idx];
    }
    // Base Agent reset() is marked virtual, hence we expect derived Agent's reset to be called here
    agent->reset({start_pos_x, start_pos_y}, race_track_->headings_[reset_idx] + heading_offset);
}

// Iterates 1 step in the environment given the current action and returns the next state of the agent
void Environment::step()
{
    // -------- 1) Kinematics Update of Agents -------- //
    for (uint16_t i{0}; i < agents_.size(); i++)
    {
        auto &agent              = agents_[i];
        auto &displacement_stats = displacement_stats_[i];
        if (!agent->crashed_)
        {
            agent->move();
            checkAndUpdateStandstill(*agent, displacement_stats);
            if (displacement_stats.displacement_timed_out)
            {
                agent->crashed_   = true;
                agent->timed_out_ = true;
            }
        }
    }

    // -------- 2) Collision detection -------- //
    collision_checker_->checkCollision();

    // -------- 3) Vector-based rendering -------- //
    visualizer_->render(*race_track_, agents_, draw_rays_ ? collision_checker_.get() : nullptr);
}

bool Environment::isEnterPressed() const
{
    struct termios oldt, newt;
    char           ch;
    tcgetattr(STDIN_FILENO, &oldt); // Get current terminal settings
    newt = oldt;
    newt.c_lflag &= ~ICANON; // Disable canonical mode
    newt.c_lflag &= ~ECHO;   // Disable echoing
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    // Set the terminal to non-blocking mode
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

    if (read(STDIN_FILENO, &ch, 1) == 1)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // Restore terminal settings
        return ch == '\n';
    }
    else
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // Restore terminal settings
    }
    return false; // No key was pressed
}

void Environment::saveImage(const std::string &filename) const
{
    screen_grabber_->saveRenderTargetToFile(filename);
}
