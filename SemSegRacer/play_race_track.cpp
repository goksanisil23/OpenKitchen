#include <iostream>
#include <string>
#include <vector>

#include "raylib-cpp.hpp"
#include "raylib_utils.hpp"
#include "utils.hpp"

struct CurImg
{
};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a folder path for trace track csv files\n";
        return -1;
    }

    std::string              race_track_dir(argv[1]);
    std::vector<std::string> track_csv_files{race_track_gen::getCSVFilesInDirectory(race_track_dir)};

    const auto file = track_csv_files[0];
    // for (const auto &file : track_csv_files)
    {
        race_track_gen::TrackData track_data_points;
        race_track_gen::getTrackDataFromCsv(file, track_data_points);

        const auto track_extent = race_track_gen::calculateTrackExtents(track_data_points);
        race_track_gen::centerTrackPointsToWindow(
            track_extent, okitch::kScreenWidth, okitch::kScreenHeight, track_data_points);
        const size_t num_pts{track_data_points.x_m.size()};

        // Calculate track boundaries and lane centers
        std::vector<float> left_bound_inner_x, left_bound_inner_y, left_bound_outer_x, left_bound_outer_y,
            right_bound_inner_x, right_bound_inner_y, right_bound_outer_x, right_bound_outer_y;
        race_track_gen::calculateTrackLanes(track_data_points,
                                            left_bound_inner_x,
                                            left_bound_inner_y,
                                            left_bound_outer_x,
                                            left_bound_outer_y,
                                            right_bound_inner_x,
                                            right_bound_inner_y,
                                            right_bound_outer_x,
                                            right_bound_outer_y);

        // Visualization data
        raylib::Vector2 *right_bnd_inner_pts = okitch::convertToRaylibArray(right_bound_inner_x, right_bound_inner_y);
        raylib::Vector2 *left_bnd_inner_pts  = okitch::convertToRaylibArray(left_bound_inner_x, left_bound_inner_y);
        raylib::Vector2 *right_bnd_outer_pts = okitch::convertToRaylibArray(right_bound_outer_x, right_bound_outer_y);
        raylib::Vector2 *left_bnd_outer_pts  = okitch::convertToRaylibArray(left_bound_outer_x, left_bound_outer_y);

        okitch::Vis visualizer(okitch::kScreenWidth, okitch::kScreenHeight);
        visualizer.enableImageSharing();

        okitch::Driver driver({track_data_points.x_m[0], track_data_points.y_m[0]}, 0.F);

        while (!visualizer.window_->ShouldClose())
        {
            driver.update();
            visualizer.activateDrawing(driver.pos_, driver.rot_);
            {
                okitch::shadeAreaBetweenCurves(
                    right_bnd_inner_pts, right_bnd_outer_pts, num_pts, raylib::Color(0, 0, 255));
                okitch::shadeAreaBetweenCurves(
                    left_bnd_inner_pts, left_bnd_outer_pts, num_pts, raylib::Color(255, 0, 0));
                visualizer.drawDriver(driver.pos_, driver.rot_);
            }
            visualizer.deactivateDrawing();
        }
        visualizer.window_->Close();

        delete[] right_bnd_inner_pts;
        delete[] left_bnd_inner_pts;
        delete[] right_bnd_outer_pts;
        delete[] left_bnd_outer_pts;
    }

    return 0;
}