#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "raylib_msgs.h"

namespace evo_driver
{
std::random_device                    rd;
std::mt19937                          rand_engine(rd());
std::uniform_real_distribution<float> rand_dist(0.0F, 1.0F);

class EvoController
{
  public:
    struct EvoControls
    {
        bool accelerate, decelerate, turn_left, turn_right;
    };

    void update(const std::vector<okitch::Vec2d> &sensor_meas)
    {
        if (sensor_meas.size() > 20)
        {
            controls.accelerate = true;
        }
        if (sensor_meas.size() < 10)
        {
            controls.turn_left = true;
        }
    }

    EvoControls controls;
};

} // namespace evo_driver