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

    constexpr int kScreenWidth  = 1600;
    constexpr int kScreenHeight = 1000;

    for (const auto &file : track_csv_files)
    {
        race_track_gen::TrackData track_data_points;
        race_track_gen::getTrackDataFromCsv(file, track_data_points);

        const auto track_extent = race_track_gen::calculateTrackExtents(track_data_points);
        race_track_gen::centerTrackPointsToWindow(track_extent, kScreenWidth, kScreenHeight, track_data_points);
        const size_t num_pts{track_data_points.x_m.size()};

        // Calculate track boundaries
        std::vector<float> right_bound_x, right_bound_y, left_bound_x, left_bound_y;
        calculateTrackLanes(track_data_points, right_bound_x, right_bound_y, left_bound_x, left_bound_y);

        // Visualization data
        raylib::Vector2 *center_points = okitch::convertToRaylibArray(track_data_points.x_m, track_data_points.y_m);
        raylib::Vector2 *right_bnd_pts = okitch::convertToRaylibArray(right_bound_x, right_bound_y);
        raylib::Vector2 *left_bnd_pts  = okitch::convertToRaylibArray(left_bound_x, left_bound_y);

        okitch::Vis visualizer(kScreenWidth, kScreenHeight);

        raylib::Color center_color(raylib::Color::White());
        raylib::Color right_bound_color(raylib::Color::Red());
        raylib::Color left_bound_color(raylib::Color::Green());
        raylib::Color left_lane_color(raylib::Color::Yellow());
        while (!visualizer.window_->ShouldClose())
        {
            visualizer.activateDrawing();
            {
                okitch::shadeAreaBetweenCurves(center_points, right_bnd_pts, num_pts, raylib::Color::Yellow());
                okitch::shadeAreaBetweenCurves(center_points, left_bnd_pts, num_pts, raylib::Color::Red());
            }
            visualizer.deactivateDrawing();
        }

        visualizer.window_->Close();

        delete[] center_points;
        delete[] right_bnd_pts;
        delete[] left_bnd_pts;
    }

    return 0;
}