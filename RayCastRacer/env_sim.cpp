#include <iostream>
#include <string>
#include <vector>

#include "race_track_utils.hpp"
#include "raylib-cpp.hpp"
#include "raylib_utils.hpp"
#include "spmc_queue.h"

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

    okitch::Driver driver({track_data_points.x_m[0], track_data_points.y_m[0]}, 0.F);
    driver.enableRaycastSensor();

    okitch::Vis visualizer(okitch::kScreenWidth, okitch::kScreenHeight);

    auto sensor_msg_shm_q = shmmap<okitch::Laser2dMsg<100>, 4>("laser_msgs"); // shared memory object
    assert(sensor_msg_shm_q);

    while (!WindowShouldClose())
    {
        driver.update();
        visualizer.activateDrawing(driver);
        {
            okitch::shadeAreaBetweenCurves(right_bound_inner, right_bound_outer, raylib::Color(0, 0, 255));
            okitch::shadeAreaBetweenCurves(left_bound_inner, left_bound_outer, raylib::Color(255, 0, 0));
            okitch::shadeAreaBetweenCurves(left_bound_inner, right_bound_inner, raylib::Color(0, 255, 0));
            visualizer.drawDriver(driver);
        }
        visualizer.disableDrawing();

        std::vector<okitch::Vec2d> sensor_hits;
        // This section is for direct render buffer manipulation
        {
            raylib::Image render_buffer;
            render_buffer.Load(visualizer.render_target_.texture);
            driver.updateSensor(render_buffer, sensor_hits);
            UpdateTexture(visualizer.render_target_.texture, render_buffer.data);
        }
        std::cout << sensor_hits.size() << std::endl;

        // Render the final texture
        visualizer.render();

        // Send the sensor readings over shared mem
        sensor_msg_shm_q->write(
            [&sensor_hits](okitch::Laser2dMsg<100> &msg)
            {
                msg.size = sensor_hits.size();
                for (size_t i{0}; i < sensor_hits.size(); i++)
                {
                    msg.data[i] = sensor_hits[i];
                }
            });
    }

    return 0;
}