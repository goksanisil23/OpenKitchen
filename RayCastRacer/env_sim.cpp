#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "EvoDriver.hpp"
#include "race_track_utils.hpp"
#include "raylib-cpp.hpp"
#include "raylib_utils.hpp"
#include "spmc_queue.h"

constexpr int16_t kNumDrivers{25};
constexpr int16_t kNumGenerations{100}; // number of generations
constexpr size_t  kStartingIdx{1};      // along which point on the track to start

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

// Given a 2d-coordinate, finds the nearest track-center coordinate index
size_t findNearestTrackIndexBruteForce(const race_track_gen::TrackData &track_data_points,
                                       const okitch::Vec2d             &query_pt)
{
    float  min_distance = std::numeric_limits<float>::max();
    size_t min_idx{0};
    float  distance;
    for (size_t i{0}; i < track_data_points.x_m.size(); i++)
    {
        distance = query_pt.distanceSquared({track_data_points.x_m[i], track_data_points.y_m[i]});
        if (distance < min_distance)
        {
            min_distance = distance;
            min_idx      = i;
        }
    }
    return min_idx;
}

void assignScores(const std::vector<okitch::Driver>      &drivers,
                  const race_track_gen::TrackData        &track_data_points,
                  std::vector<evo_driver::EvoController> &driver_controllers)
{
    for (size_t i{0}; i < drivers.size(); i++)
    {
        auto nearest_track_idx =
            findNearestTrackIndexBruteForce(track_data_points, {drivers[i].pos_.x, drivers[i].pos_.y});
        driver_controllers[i].score_ = static_cast<float>(nearest_track_idx);
    }
}

// Draws actuation bars on the top right corner
void drawActionBar(const std::vector<evo_driver::EvoController> &driver_controllers,
                   const std::vector<okitch::Driver>            &drivers,
                   const uint32_t                                iter)
{
    constexpr int32_t kTotalBarLen(200);
    constexpr int32_t kBarStartX   = okitch::kScreenWidth - kTotalBarLen;
    constexpr int32_t kBarStartY   = 20;
    constexpr int32_t kBarSpacingY = 10;
    DrawText(TextFormat("Accel. (iter:%u)", iter), kBarStartX + kTotalBarLen / 2 - 30, 0, 11, LIGHTGRAY);
    for (size_t i{0}; i < driver_controllers.size(); i++)
    {
        DrawText(TextFormat("%lu", i), kBarStartX - 30, kBarStartY + i * kBarSpacingY, 10, WHITE);
        okitch::drawLoadBar(kBarStartX,
                            kBarStartY + i * kBarSpacingY,
                            driver_controllers[i].controls_.acceleration_delta,
                            kTotalBarLen,
                            !drivers[i].crashed_);
    }
}

// Draws the average colony score per episode
void showColonyScore(const std::vector<float> &colony_avg_scores)
{
    std::cout << "**** avg colony scores per episode ****" << std::endl;
    for (size_t i{0}; i < colony_avg_scores.size(); i++)
    {
        printf("%.2f ", colony_avg_scores[i]);
    }
    std::cout << std::endl;
}

void drawTrackPointNumbers(const race_track_gen::TrackData &track_data_points)
{
    for (size_t i{0}; i < track_data_points.x_m.size(); i++)
    {
        DrawText(TextFormat("%lu", i), track_data_points.x_m[i], track_data_points.y_m[i], 10, BLACK);
    }
}

float getAvgColonyScore(const std::vector<evo_driver::EvoController> &driver_controllers)
{
    // Calculate the current average
    float current_avg{0.F};
    for (const auto &agent : driver_controllers)
    {
        current_avg += agent.score_;
    }
    current_avg /= static_cast<float>(driver_controllers.size());

    return current_avg;
}

void saveBestAgentNetwork(const std::vector<evo_driver::EvoController> &driver_controllers)
{
    static float top_score_all_time{0.F};  // individual top score of an agent across all generations
    static float prev_gen_best_score{0.F}; // keeping track of the previous generation's best score
    size_t       top_scorer_agent_id;
    float        best_score_current{0.F};
    for (size_t i{0}; i < driver_controllers.size(); i++)
    {

        if (driver_controllers[i].score_ > best_score_current)
        {
            best_score_current  = driver_controllers[i].score_;
            top_scorer_agent_id = i;
        }
    }

    if (prev_gen_best_score > best_score_current)
    {
        std::string error_string{"Current gen. high score " + std::to_string(best_score_current) +
                                 "is less than previous gen. high score " + std::to_string(prev_gen_best_score)};
        // throw std::runtime_error(error_string);
        std::cerr << error_string << std::endl;
    }
    prev_gen_best_score = best_score_current;

    if (best_score_current > top_score_all_time)
    {
        // Save the weights
        evo_driver::writeMatrixToFile("agent_weights_1.txt", driver_controllers[top_scorer_agent_id].nn_.weights_1_);
        evo_driver::writeMatrixToFile("agent_weights_2.txt", driver_controllers[top_scorer_agent_id].nn_.weights_2_);

        top_score_all_time = best_score_current;
    }
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
    float starting_angle_deg;

    const auto track_extent = race_track_gen::calculateTrackExtents(track_data_points);
    // Calculate track boundaries and lane centers
    std::vector<okitch::Vec2d> left_bound_inner, left_bound_outer, right_bound_inner, right_bound_outer;
    race_track_gen::centerTrackPointsToWindow(
        track_extent, okitch::kScreenWidth, okitch::kScreenHeight, track_data_points);
    race_track_gen::calculateTrackLanes(track_data_points,
                                        left_bound_inner,
                                        left_bound_outer,
                                        right_bound_inner,
                                        right_bound_outer,
                                        starting_angle_deg);
    std::vector<okitch::Vec2d> start_line{right_bound_outer.front(), right_bound_outer.back()};
    std::vector<okitch::Vec2d> finish_line{left_bound_outer.front(), left_bound_outer.back()};

    okitch::Vis visualizer;

    std::vector<okitch::Driver> drivers;
    for (int16_t i{0}; i < kNumDrivers; i++)
    {
        drivers.push_back(okitch::Driver{{track_data_points.x_m[kStartingIdx], track_data_points.y_m[kStartingIdx]},
                                         starting_angle_deg,
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

    uint32_t           episode_idx{0};
    uint32_t           iter{0};
    std::vector<float> colony_avg_scores;
    while (!WindowShouldClose())
    {
        if (reset_generation)
        {
            std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
            iter = 0;
            // First, assign scores based on how far along the track the agents has come
            assignScores(drivers, track_data_points, driver_controllers);
            saveBestAgentNetwork(driver_controllers);
            colony_avg_scores.emplace_back(getAvgColonyScore(driver_controllers));
            showColonyScore(colony_avg_scores);
            episode_idx++;
            // Mate the drivers before resetting
            evo_driver::chooseAndMateAgents(driver_controllers);

            for (int16_t i{0}; i < kNumDrivers; i++)
            {
                drivers[i].reset({track_data_points.x_m[kStartingIdx], track_data_points.y_m[kStartingIdx]},
                                 starting_angle_deg);
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
                    driver_controllers[i].score_ = 1; // minimum non-zero score
                    drivers[i].crashed_          = true;
                }
            }
        }
        visualizer.activateDrawing();
        {
            okitch::shadeAreaBetweenCurves(right_bound_inner, right_bound_outer, raylib::Color(0, 0, 255, 255));
            okitch::shadeAreaBetweenCurves(left_bound_inner, left_bound_outer, raylib::Color(255, 0, 0, 255));
            okitch::shadeAreaBetweenCurves(left_bound_inner, right_bound_inner, raylib::Color(0, 255, 0, 255));
            okitch::shadeAreaBetweenCurves(start_line, finish_line, raylib::Color(0, 0, 255, 255));
            // drawTrackPointNumbers(track_data_points);
            drawActionBar(driver_controllers, drivers, iter);
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
        iter++;

        // Send the sensor readings over shared mem
        sensor_msg_shm_q->write(
            [&sensor_hits](okitch::Laser2dMsg<100> &msg)
            {
                msg.size = sensor_hits[0].size();
                for (size_t i{0}; i < sensor_hits[0].size(); i++)
                {
                    msg.data[i] = sensor_hits[0][i];
                }
            });
    }

    return 0;
}