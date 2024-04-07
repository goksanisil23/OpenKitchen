#pragma once
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Agent.h"
#include "Environment/RaceTrack.hpp"
#include "Environment/Visualizer.h"

#include "raylib-cpp.hpp"

namespace rl
{
class Environment
{
  public:
    struct State
    {
        float              speed;
        float              rotation;
        std::vector<Vec2d> sensor_hits;
    };

    Environment(const std::string &race_track_path)
    {
        race_track_ = std::make_unique<RaceTrack>(race_track_path);
        visualizer_ = std::make_unique<env::Visualizer>();
    }

    void setAgent(Agent *agent)
    {
        agents_.push_back(agent);
    }

    void drawSensorRanges(const std::vector<Vec2d> &sensor_hits)
    {
        std::string range_string{""};
        for (const auto &hit : sensor_hits)
        {
            range_string += std::to_string(static_cast<int64_t>(hit.norm())) + " ";
        }
        DrawText(range_string.c_str(), kScreenWidth - 400, 100, 15, WHITE);
    }

    // Iterates 1 step in the environment given the current action and returns the next state of the agent
    void step()
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
            render_buffer_ = std::make_unique<raylib::Image>();
            render_buffer_->Load(visualizer_->render_target_.texture);
            // We first check the collision before drawing any sensor or agents to avoid overlap
            // NOTE: Sensor update needs to happen before drawing multiple agents since we emulate parallel simulators here so agents
            // should NOT see each other's world.
            for (auto &agent : agents_)
            {
                if (!agent->crashed_)
                {
                    if (visualizer_->checkAgentCollision(*render_buffer_, *agent))
                    {
                        agent->crashed_ = true;
                    }
                    else
                    {
                        visualizer_->updateSensor(*agent, *render_buffer_);
                    }
                }
            }
            for (auto &agent : agents_)
            {
                visualizer_->drawAgent(*agent, *render_buffer_);
            }
            UpdateTexture(visualizer_->render_target_.texture, render_buffer_->data);
        }
        // -------- 4) Render the final texture -------- //
        visualizer_->render();
    }

  public:
    std::unique_ptr<RaceTrack>       race_track_;
    std::unique_ptr<env::Visualizer> visualizer_;
    std::vector<Agent *>             agents_;
    std::unique_ptr<raylib::Image>   render_buffer_{nullptr};
};
} // namespace rl