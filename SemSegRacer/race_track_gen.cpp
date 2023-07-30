#include <iostream>
#include <string>
#include <vector>

#include "raylib-cpp.hpp"
#include "raylib_utils.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a folder path for trace track csv files\n";
        return -1;
    }

    std::string              race_track_dir(argv[1]);
    std::vector<std::string> track_csv_files{race_track_gen::getCSVFilesInDirectory(race_track_dir)};

    constexpr int kScreenWidth  = 1000;
    constexpr int kScreenHeight = 1000;

    for (const auto &file : track_csv_files)
    {
        race_track_gen::TrackData track_data_points;
        race_track_gen::getTrackDataFromCsv(file, track_data_points);

        const auto track_extent = race_track_gen::calculateTrackExtents(track_data_points);
        race_track_gen::centerTrackPointsToWindow(track_extent, kScreenWidth, kScreenHeight, track_data_points);
        const size_t num_pts{track_data_points.x_m.size()};

        // Calculate track boundaries and lane centers
        std::vector<float> right_bound_x, right_bound_y, left_bound_x, left_bound_y, left_lane_center_x,
            left_lane_center_y, right_lane_center_x, right_lane_center_y;
        race_track_gen::calculateTrackLanes(track_data_points,
                                            right_bound_x,
                                            right_bound_y,
                                            left_bound_x,
                                            left_bound_y,
                                            left_lane_center_x,
                                            left_lane_center_y,
                                            right_lane_center_x,
                                            right_lane_center_y);

        // Visualization data
        raylib::Vector2 *center_points = okitch::convertToRaylibArray(track_data_points.x_m, track_data_points.y_m);
        raylib::Vector2 *right_bnd_pts = okitch::convertToRaylibArray(right_bound_x, right_bound_y);
        raylib::Vector2 *left_bnd_pts  = okitch::convertToRaylibArray(left_bound_x, left_bound_y);
        raylib::Vector2 *left_lane_center_pts  = okitch::convertToRaylibArray(left_lane_center_x, left_lane_center_y);
        raylib::Vector2 *right_lane_center_pts = okitch::convertToRaylibArray(right_lane_center_x, right_lane_center_y);

        okitch::Vis    visualizer(kScreenWidth, kScreenHeight);
        okitch::Driver driver({track_data_points.x_m[0], track_data_points.y_m[0]}, 0.F);

        // while (!visualizer.window_->ShouldClose())
        for (size_t i{0}; i < left_lane_center_x.size(); i++)
        {
            driver.update();
            driver.pos_.x = left_lane_center_x[i];
            driver.pos_.y = left_lane_center_y[i];
            visualizer.activateDrawing(driver.pos_, driver.rot_);
            {
                okitch::shadeAreaBetweenCurves(center_points, right_bnd_pts, num_pts, raylib::Color(0, 0, 255));
                okitch::shadeAreaBetweenCurves(center_points, left_bnd_pts, num_pts, raylib::Color(255, 0, 0));
                // DrawLineStrip(left_lane_center_pts, left_lane_center_x.size(), raylib::Color::Yellow());
                // DrawLineStrip(right_lane_center_pts, right_lane_center_x.size(), raylib::Color::Yellow());
                // visualizer.drawDriver(driver.pos_, driver.rot_);
            }
            visualizer.deactivateDrawing();
            if (i % 10 == 0)
                visualizer.saveImage();
        }
        // visualizer.saveImage();
        visualizer.window_->Close();

        delete[] center_points;
        delete[] right_bnd_pts;
        delete[] left_bnd_pts;
    }

    return 0;
}