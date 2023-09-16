#include <iostream>
#include <string>
#include <vector>

#include "EvoDriver.hpp"
#include "race_track_utils.hpp"
#include "raylib-cpp.hpp"
#include "raylib_utils.hpp"
#include "spmc_queue.h"

constexpr int16_t kNumDrivers{3};
constexpr int16_t kNumGenerations{100}; // number of generations
constexpr size_t  kStartingIdx{0};      // along which point on the track to start

bool shouldResetEpisode(const std::vector<okitch::Driver> &drivers)
{
    size_t num_drv_crashed{0};
    for (const auto &driver : drivers)
    {
        if (driver.crashed_)
        {
            num_drv_crashed++;
        }
    }
    if (num_drv_crashed == drivers.size())
    {
        return true;
    }
    return false;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    race_track_gen::TrackData track_data_points;
    race_track_gen::getTrackDataFromCsv(argv[1], track_data_points);

    const auto track_extent = race_track_gen::calculateTrackExtents(track_data_points);
    // Calculate track boundaries and lane centers
    std::vector<race_track_gen::Vec2> left_bound_inner, left_bound_outer, right_bound_inner, right_bound_outer;
    race_track_gen::centerTrackPointsToWindow(
        track_extent, okitch::kScreenWidth, okitch::kScreenHeight, track_data_points);
    race_track_gen::calculateTrackLanes(
        track_data_points, left_bound_inner, left_bound_outer, right_bound_inner, right_bound_outer);

    okitch::Vis visualizer;

    std::vector<okitch::Driver> drivers;
    for (int16_t i{0}; i < kNumDrivers; i++)
    {
        drivers.push_back(okitch::Driver{{track_data_points.x_m[kStartingIdx], track_data_points.y_m[kStartingIdx]},
                                         -90.F,
                                         i,
                                         visualizer.camera_.get() != nullptr});
        drivers.back().enableRaycastSensor();
        drivers.back().enableAutoControl();
        // drivers.back().enableManualControl();
    }

    std::vector<std::vector<okitch::Vec2d>> sensor_hits(drivers.size());
    std::vector<evo_driver::EvoController>  driver_controllers(drivers.size());
    auto sensor_msg_shm_q = shmmap<okitch::Laser2dMsg<100>, 4>("laser_msgs"); // shared memory object
    assert(sensor_msg_shm_q);

    bool reset_generation = false;

    uint32_t episode_idx{0};
    while (!WindowShouldClose())
    {
        if (reset_generation)
        {
            std::cout << "------------ EPISODE " << episode_idx << " ---------------" << std::endl;
            episode_idx++;
            // Mate the drivers before resetting
            evo_driver::chooseAndMateAgents(driver_controllers);

            for (int16_t i{0}; i < kNumDrivers; i++)
            {
                drivers[i].reset({track_data_points.x_m[kStartingIdx], track_data_points.y_m[kStartingIdx]}, -90.F);
            }

            reset_generation = false;
        }

        for (int16_t i{0}; i < kNumDrivers; i++)
        {
            if (!drivers[i].crashed_)
            {
                // drivers[i].updateManualControl();
                drivers[i].updateAutoControl(driver_controllers[i].controls_.acceleration_delta,
                                             driver_controllers[i].controls_.steering_delta);
                // increase the score for every step of the episode agent has not crashed
                if (drivers[i].standstill_timed_out_)
                {
                    driver_controllers[i].score_ = 0;
                    drivers[i].crashed_          = true;
                }
                else
                {
                    driver_controllers[i].score_++;
                }
            }
        }
        visualizer.activateDrawing();
        {
            okitch::shadeAreaBetweenCurves(right_bound_inner, right_bound_outer, raylib::Color(0, 0, 255, 255));
            okitch::shadeAreaBetweenCurves(left_bound_inner, left_bound_outer, raylib::Color(255, 0, 0, 255));
            okitch::shadeAreaBetweenCurves(left_bound_inner, right_bound_inner, raylib::Color(0, 255, 0, 255));
        }
        visualizer.disableDrawing();

        // This section is for direct render buffer manipulation
        {
            raylib::Image render_buffer;
            render_buffer.Load(visualizer.render_target_.texture);
            // We first check the collision before drawing any sensor or drivers to avoid overlap
            for (auto &driver : drivers)
            {
                if (!driver.crashed_)
                    driver.crashed_ = visualizer.checkDriverCollision(render_buffer, driver);
            }
            for (size_t i{0}; i < drivers.size(); i++)
            {
                visualizer.drawDriver(drivers[i], render_buffer);
                drivers[i].updateSensor(render_buffer, sensor_hits[i]);
            }
            UpdateTexture(visualizer.render_target_.texture, render_buffer.data);
        }

        // Render the final texture
        visualizer.render();

        for (int16_t i{0}; i < kNumDrivers; i++)
        {
            driver_controllers[i].updateAction(drivers[i].speed_, drivers[i].rot_, sensor_hits[i]);
        }

        // If all the robots have crashed, reset the generation
        reset_generation = shouldResetEpisode(drivers);

        // Send the sensor readings over shared mem
        // sensor_msg_shm_q->write(
        //     [&sensor_hits](okitch::Laser2dMsg<100> &msg)
        //     {
        //         msg.size = sensor_hits.size();
        //         for (size_t i{0}; i < sensor_hits.size(); i++)
        //         {
        //             msg.data[i] = sensor_hits[i];
        //         }
        //     });
    }

    return 0;
}