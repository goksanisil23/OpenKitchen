#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
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

struct Vec2
{
    float x;
    float y;
};
struct Extent2d
{
    float min_x;
    float min_y;
    float max_x;
    float max_y;

    bool isPointInside(const Vec2 &pt)
    {
        if ((pt.x > min_x) && (pt.y > min_y) && (pt.x < max_x) && (pt.y < max_y))
            return true;
        return false;
    }
};

std::ostream &operator<<(std::ostream &os, const Extent2d &obj)
{
    os << obj.min_x << " " << obj.min_y << " " << obj.max_x << " " << obj.max_y << std::endl;
    return os;
}

struct TrackTile
{
    std::vector<Vec2> left_bound_inner;
    std::vector<Vec2> left_bound_outer;
    std::vector<Vec2> right_bound_inner;
    std::vector<Vec2> right_bound_outer;
    Extent2d          tile_extent;
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
    const float track_width    = track_extent.max_x - track_extent.min_x;
    const float track_height   = track_extent.max_y - track_extent.min_y;
    float       scale_factor_x = window_width / track_width;
    float       scale_factor_y = window_height / track_height;
    float       scale_factor   = std::min(scale_factor_x, scale_factor_y);
    // Scale down a bit more to have some padding on the screen
    constexpr float kScreenFitScale{0.9};
    scale_factor *= kScreenFitScale;

    for (size_t i{}; i < track_data_points_inout.x_m.size(); i++)
    {
        track_data_points_inout.x_m[i] *= scale_factor;
        track_data_points_inout.y_m[i] *= scale_factor;
        track_data_points_inout.w_tr_left_m[i] *= scale_factor;
        track_data_points_inout.w_tr_right_m[i] *= scale_factor;
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

void updateExtent(const std::vector<Vec2> &vec, Extent2d &extent)
{
    for (size_t i{0}; i < vec.size(); i++)
    {
        const auto pt_x = vec[i].x;
        const auto pt_y = vec[i].y;
        if (pt_x < extent.min_x)
        {
            extent.min_x = pt_x;
        }
        if (pt_x > extent.max_x)
        {
            extent.max_x = pt_x;
        }
        if (pt_y < extent.min_y)
        {
            extent.min_y = pt_y;
        }
        if (pt_y > extent.max_y)
        {
            extent.max_y = pt_y;
        }
    }
}

std::vector<TrackTile> divideBoundsIntoTiles(const std::vector<Vec2> &left_bound_inner,
                                             const std::vector<Vec2> &left_bound_outer,
                                             const std::vector<Vec2> &right_bound_inner,
                                             const std::vector<Vec2> &right_bound_outer)
{
    // Find the extents first
    Extent2d track_extent{std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max()};

    updateExtent(left_bound_inner, track_extent);
    updateExtent(left_bound_outer, track_extent);
    updateExtent(right_bound_inner, track_extent);
    updateExtent(right_bound_outer, track_extent);

    std::cout << "track extent: " << track_extent << std::endl;

    // Divide the bounding extent into tiles
    constexpr float kTileLength(200.); // 50 meter tiles
    size_t          num_tiles_x = std::ceil((track_extent.max_x - track_extent.min_x) / kTileLength);
    size_t          num_tiles_y = std::ceil((track_extent.max_y - track_extent.min_y) / kTileLength);

    // Find the indices of bound points that lie within a tile
    std::vector<TrackTile> track_tiles(num_tiles_x * num_tiles_y);
    for (size_t tile_idx_y{0}; tile_idx_y < num_tiles_y; tile_idx_y++)
    {
        for (size_t tile_idx_x{0}; tile_idx_x < num_tiles_x; tile_idx_x++)
        {
            size_t tile_idx = tile_idx_x + num_tiles_x * tile_idx_y;
            // Bounding box of the current tile
            float    top_left_x = track_extent.min_x + static_cast<float>(tile_idx_x) * kTileLength;
            float    top_left_y = track_extent.min_y + static_cast<float>(tile_idx_y) * kTileLength;
            Extent2d tile_extent{top_left_x, top_left_y, top_left_x + kTileLength, top_left_y + kTileLength};
            track_tiles.at(tile_idx).tile_extent = tile_extent;

            // std::cout << " tile: " << tile_extent << std::endl;

            for (size_t pt_idx{0}; pt_idx < left_bound_inner.size(); pt_idx++)
            {
                // All boundaries on the same index must fall within the same tile
                if (tile_extent.isPointInside(left_bound_inner[pt_idx]) &&
                    tile_extent.isPointInside(left_bound_outer[pt_idx]) &&
                    tile_extent.isPointInside(right_bound_inner[pt_idx]) &&
                    tile_extent.isPointInside(right_bound_outer[pt_idx]))
                {
                    track_tiles.at(tile_idx).left_bound_inner.push_back(left_bound_inner[pt_idx]);
                    track_tiles.at(tile_idx).left_bound_outer.push_back(left_bound_outer[pt_idx]);
                    track_tiles.at(tile_idx).right_bound_inner.push_back(right_bound_inner[pt_idx]);
                    track_tiles.at(tile_idx).right_bound_outer.push_back(right_bound_outer[pt_idx]);
                }
            }
            if (!track_tiles.at(tile_idx).right_bound_inner.empty())
                std::cout << "tile (" << tile_idx_x << ", " << tile_idx_y
                          << "): " << track_tiles.at(tile_idx).right_bound_inner.size() << std::endl;
        }
    }

    return track_tiles;
}

// Given the track center coordinates and left & right lane widths, calculates the track boundaries on left and right
void calculateTrackLanes(const TrackData   &track_data_points,
                         std::vector<Vec2> &left_bound_inner,
                         std::vector<Vec2> &left_bound_outer,
                         std::vector<Vec2> &right_bound_inner,
                         std::vector<Vec2> &right_bound_outer)
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

    left_bound_inner.resize(dx.size());
    left_bound_outer.resize(dx.size());
    right_bound_inner.resize(dx.size());
    right_bound_outer.resize(dx.size());

    constexpr float kBoundaryThickness{5.F};

    // Calculate track boundaries that are perpendicular lane width distance away from track center
    for (size_t i = 0; i < dx.size(); ++i)
    {
        right_bound_inner[i].x = track_data_points.x_m[i] + track_data_points.w_tr_right_m[i] * dy[i];
        right_bound_inner[i].y = track_data_points.y_m[i] - track_data_points.w_tr_right_m[i] * dx[i];

        left_bound_inner[i].x = track_data_points.x_m[i] - track_data_points.w_tr_left_m[i] * dy[i];
        left_bound_inner[i].y = track_data_points.y_m[i] + track_data_points.w_tr_left_m[i] * dx[i];

        right_bound_outer[i].x =
            track_data_points.x_m[i] + (track_data_points.w_tr_right_m[i] + kBoundaryThickness) * dy[i];
        right_bound_outer[i].y =
            track_data_points.y_m[i] - (track_data_points.w_tr_right_m[i] + kBoundaryThickness) * dx[i];

        left_bound_outer[i].x =
            track_data_points.x_m[i] - (track_data_points.w_tr_left_m[i] + kBoundaryThickness) * dy[i];
        left_bound_outer[i].y =
            track_data_points.y_m[i] + (track_data_points.w_tr_left_m[i] + kBoundaryThickness) * dx[i];
    }
}

} // namespace race_track_gen
