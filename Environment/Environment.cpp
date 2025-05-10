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

Environment::Environment(const std::string &race_track_path, const std::vector<Agent *> &agents)
{
    race_track_ = std::make_unique<RaceTrack>(race_track_path);
    visualizer_ = std::make_unique<env::Visualizer>();

    // Initialize the collision checker with the framebuffer object ID
    collision_checker_ = std::make_unique<CollisionChecker>(visualizer_->render_target_.texture.id, agents);

    // Store pointers to the agents
    agents_ = agents;
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

void Environment::resetAgent(Agent *agent, const bool pick_random_point)
{
    int32_t     reset_idx = pick_random_point ? pickRandomResetTrackIdx() : RaceTrack::kStartingIdx;
    const float start_pos_x{race_track_->track_data_points_.x_m[reset_idx]};
    const float start_pos_y{race_track_->track_data_points_.y_m[reset_idx]};
    // Base Agent reset() is marked virtual, hence we expect derived Agent's reset to be called here
    agent->reset({start_pos_x, start_pos_y}, race_track_->headings_[reset_idx]);
}

// Iterates 1 step in the environment given the current action and returns the next state of the agent
void Environment::step()
{
    // -------- 1) Kinematics Update of Agents -------- //
    for (auto agent : agents_)
    {
        if (!agent->crashed_)
        {
            agent->move();
            if (agent->standstill_timed_out_)
            {
                agent->crashed_ = true;
            }
        }
    }

    // -------- 2) Drawing static parts of the environment -------- //
    visualizer_->activateDrawing();
    {
        env::Visualizer::shadeAreaBetweenCurves(
            race_track_->right_bound_inner_, race_track_->right_bound_outer_, raylib::Color(0, 0, 255, 255));
        env::Visualizer::shadeAreaBetweenCurves(
            race_track_->left_bound_inner_, race_track_->left_bound_outer_, raylib::Color(255, 0, 0, 255));
        env::Visualizer::shadeAreaBetweenCurves(
            race_track_->left_bound_inner_, race_track_->right_bound_inner_, raylib::Color(0, 255, 0, 255));
        // env::Visualizer::shadeAreaBetweenCurves(
        // race_track_->start_line_, race_track_->finish_line_, raylib::Color(0, 0, 255, 255));

        // For continuous loop
        if constexpr (env::kContinuousLoop)
        {
            env::Visualizer::shadeAreaBetweenCurves(
                race_track_->start_line_, race_track_->finish_line_, raylib::Color(0, 255, 0, 255));
            env::Visualizer::shadeAreaBetweenCurves(
                std::vector<Vec2d>{race_track_->right_bound_inner_.front(), race_track_->right_bound_inner_.back()},
                std::vector<Vec2d>{race_track_->right_bound_outer_.front(), race_track_->right_bound_outer_.back()},
                raylib::Color(0, 0, 255, 255));
            env::Visualizer::shadeAreaBetweenCurves(
                std::vector<Vec2d>{race_track_->left_bound_inner_.front(), race_track_->left_bound_inner_.back()},
                std::vector<Vec2d>{race_track_->left_bound_outer_.front(), race_track_->left_bound_outer_.back()},
                raylib::Color(255, 0, 0, 255));
        }
        else
        {
            env::Visualizer::shadeAreaBetweenCurves(
                race_track_->start_line_, race_track_->finish_line_, raylib::Color(0, 255, 0, 255));
        }

        // drawTrackPointNumbers(track_data_points);
        // genetic::util::drawActionBar(agents, iteration);
        // genetic::util::drawEpisodeNum(episode_idx);
        // genetic::util::drawTrackTitle(race_track->track_name_);
        // visualizer_->drawTrackTitle(race_track_->track_name_);
    }
    visualizer_->disableDrawing();

    // -------- 3) Direct render buffer manipulation for sensors -------- //
    {
        // We first check the collision before drawing any sensor or agents to avoid overlap
        // NOTE: Sensor update needs to happen before drawing multiple agents since we emulate parallel simulators here so agents
        // should NOT see each other's world.
        collision_checker_->checkCollision();

        visualizer_->activateDrawing(false);
        for (auto &agent : agents_)
        {
            visualizer_->drawAgent(*agent);
        }
    }

    visualizer_->disableDrawing();

    // -------- 4) Render the final texture -------- //
    visualizer_->render();
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