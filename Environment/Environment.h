#pragma once
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Agent.h"
#include "CollisionChecker.h"
#include "RaceTrack.h"
#include "ScreenGrabber.h"
#include "Visualizer.h"

#include "raylib-cpp.hpp"

struct DisplacementStats
{
    static constexpr uint32_t kPeriod{200};
    static constexpr float    kDisplamentThreshold{20.0F};

    // If the agent has not moved sufficiently for kPeriod steps, we consider it timed out
    bool     displacement_timed_out{false};
    uint32_t displacement_ctr{0U};

    Vec2d init_pos{0.F, 0.F}; // position at the start of the period
};
class Environment
{
  public:
    Environment(const std::string &race_track_path, const std::vector<Agent *> &agents, const bool draw_rays = true);

    void drawSensorRanges(const std::vector<Vec2d> &sensor_hits);

    // Iterates 1 step in the environment given the current action and returns the next state of the agent
    void step();

    int32_t pickRandomResetTrackIdx() const;

    // If pick_random_point is true, pick a random point on the track to reset the agent
    // else reset the agent to the starting point of the track
    void resetAgent(Agent *agent, const bool pick_random_point = true);

    bool isEnterPressed() const;

    void saveImage(const std::string &filename) const;

  public:
    std::unique_ptr<RaceTrack>        race_track_;
    std::unique_ptr<env::Visualizer>  visualizer_;
    std::vector<Agent *>              agents_;
    std::vector<DisplacementStats>    displacement_stats_;
    std::unique_ptr<raylib::Image>    render_buffer_{nullptr};
    std::unique_ptr<CollisionChecker> collision_checker_{nullptr};
    std::unique_ptr<ScreenGrabber>    screen_grabber_{nullptr};
};
