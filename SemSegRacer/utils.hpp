#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace race_track_gen
{
struct TrackData
{
    std::vector<float> x_m;
    std::vector<float> y_m;
    std::vector<float> w_tr_right_m;
    std::vector<float> w_tr_left_m;
};

struct Extent2d
{
    float min_x;
    float min_y;
    float max_x;
    float max_y;
};

std::vector<std::string> getCSVFilesInDirectory(const std::string &directory_path)
{
    std::vector<std::string> csv_files;
    for (const auto &entry : std::filesystem::directory_iterator(directory_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".csv")
        {
            csv_files.push_back(entry.path().string());
        }
    }
    return csv_files;
}

// Calculates the gradient of a given point, as (h[x+1] - h[x-1])/2
void gradient(const std::vector<float> &input_vec, std::vector<float> &out_grad_vec)
{
    if (input_vec.size() <= 1)
    {
        std::cerr << "Cannot calculate gradient for a vector with less than 2 elements";
    }

    for (size_t i = 0; i < input_vec.size(); i++)
    {
        if (i == 0)
        {
            // Forward difference for the first element
            out_grad_vec.push_back(input_vec[i + 1] - input_vec[i]);
        }
        else if (i == input_vec.size() - 1)
        {
            // Backward difference for the last element
            out_grad_vec.push_back(input_vec[i] - input_vec[i - 1]);
        }
        else
        {
            // Central difference for all other elements
            out_grad_vec.push_back((input_vec[i + 1] - input_vec[i - 1]) / 2.0f);
        }
    }
}

// Extracts the track data from the original csv files that are obtained from
// https://github.com/TUMFTM/racetrack-database
void getTrackDataFromCsv(const std::string &filename, TrackData &data_points_out)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    std::string line;
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string        token;

        // Parse each token separated by commas
        std::getline(iss, token, ',');
        data_points_out.x_m.push_back(std::stof(token));

        std::getline(iss, token, ',');
        data_points_out.y_m.push_back(std::stof(token));

        std::getline(iss, token, ',');
        data_points_out.w_tr_right_m.push_back(std::stof(token));

        std::getline(iss, token, ',');
        data_points_out.w_tr_left_m.push_back(std::stof(token));
    }

    file.close();
}

// Calculates the 2D extent of a track, given the x and y coordinates of the track
Extent2d calculateTrackExtents(const TrackData &track_data_points)
{
    Extent2d track_extent{std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max()};

    for (const auto &pt_x : track_data_points.x_m)
    {
        if (pt_x < track_extent.min_x)
        {
            track_extent.min_x = pt_x;
        }
        if (pt_x > track_extent.max_x)
        {
            track_extent.max_x = pt_x;
        }
    }
    for (const auto &pt_y : track_data_points.y_m)
    {
        if (pt_y < track_extent.min_y)
        {
            track_extent.min_y = pt_y;
        }
        if (pt_y > track_extent.max_y)
        {
            track_extent.max_y = pt_y;
        }
    }
    return track_extent;
}

// Given the original track points and track extents shifts and if needed scales the track coordinates,
// such that the entire track can fit into given window size.
void centerTrackPointsToWindow(const Extent2d &track_extent,
                               const float     window_width,
                               const float     window_height,
                               TrackData      &track_data_points_inout)
{
    const float track_width  = track_extent.max_x - track_extent.min_x;
    const float track_height = track_extent.max_y - track_extent.min_y;
    // float       scale_factor_x = (track_width > window_width) ? (window_width / track_width) : 1.0;
    // float       scale_factor_y = (track_height > window_height) ? (window_height / track_height) : 1.0;
    float scale_factor_x = window_width / track_width;
    float scale_factor_y = window_height / track_height;
    // Scale down a bit more to have some padding on the screen
    constexpr float kScreenFitScale{0.9};
    scale_factor_x *= kScreenFitScale;
    scale_factor_y *= kScreenFitScale;

    for (size_t i{}; i < track_data_points_inout.x_m.size(); i++)
    {
        track_data_points_inout.x_m[i] *= scale_factor_x;
        track_data_points_inout.y_m[i] *= scale_factor_y;
    }

    // Find the extents of the scaled track
    auto const  new_extent       = calculateTrackExtents(track_data_points_inout);
    const float dist_to_center_x = (window_width / 2.F) - ((new_extent.max_x + new_extent.min_x) / 2.F);
    const float dist_to_center_y = (window_height / 2.F) - ((new_extent.max_y + new_extent.min_y) / 2.F);
    for (size_t i{}; i < track_data_points_inout.x_m.size(); i++)
    {
        track_data_points_inout.x_m[i] += dist_to_center_x;
        track_data_points_inout.y_m[i] += dist_to_center_y;
    }
}

// Given the track center coordinates and left & right lane widths, calculates the track boundaries on left and right
void calculateTrackLanes(const TrackData    &track_data_points,
                         std::vector<float> &right_bound_x,
                         std::vector<float> &right_bound_y,
                         std::vector<float> &left_bound_x,
                         std::vector<float> &left_bound_y,
                         std::vector<float> &left_lane_center_x,
                         std::vector<float> &left_lane_center_y,
                         std::vector<float> &right_lane_center_x,
                         std::vector<float> &right_lane_center_y)
{

    // Find the heading of the line so that we can draw a perpendicular point, lane width away
    std::vector<float> dx, dy;
    gradient(track_data_points.x_m, dx);
    gradient(track_data_points.y_m, dy);

    std::vector<float> mag(dx.size());
    for (size_t i = 0; i < dx.size(); ++i)
    {
        mag[i] = std::sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
    }
    // Normalize the heading vector
    for (size_t i = 0; i < dx.size(); ++i)
    {
        dx[i] /= mag[i];
        dy[i] /= mag[i];
    }

    right_bound_x.resize(dx.size());
    right_bound_y.resize(dx.size());
    left_bound_x.resize(dx.size());
    left_bound_y.resize(dx.size());
    left_lane_center_x.resize(dx.size());
    left_lane_center_y.resize(dx.size());
    right_lane_center_x.resize(dx.size());
    right_lane_center_y.resize(dx.size());

    // Calculate track boundaries that are perpendicular lane width distance away from track center
    for (size_t i = 0; i < dx.size(); ++i)
    {
        right_bound_x[i] = track_data_points.x_m[i] + track_data_points.w_tr_right_m[i] * dy[i];
        right_bound_y[i] = track_data_points.y_m[i] - track_data_points.w_tr_right_m[i] * dx[i];

        left_bound_x[i] = track_data_points.x_m[i] - track_data_points.w_tr_left_m[i] * dy[i];
        left_bound_y[i] = track_data_points.y_m[i] + track_data_points.w_tr_left_m[i] * dx[i];

        right_lane_center_x[i] = track_data_points.x_m[i] + track_data_points.w_tr_right_m[i] / 2.F * dy[i];
        right_lane_center_y[i] = track_data_points.y_m[i] - track_data_points.w_tr_right_m[i] / 2.F * dx[i];

        left_lane_center_x[i] = track_data_points.x_m[i] - track_data_points.w_tr_left_m[i] / 2.F * dy[i];
        left_lane_center_y[i] = track_data_points.y_m[i] + track_data_points.w_tr_left_m[i] / 2.F * dx[i];
    }
}

} // namespace race_track_gen
