#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>

#include "Typedefs.h"

class RaceTrack
{
  public:
    static constexpr size_t kStartingIdx{3}; // along which point on the track to start

    struct TrackData
    {
        std::vector<float> x_m;
        std::vector<float> y_m;
        std::vector<float> w_tr_right_m;
        std::vector<float> w_tr_left_m;
    };

    RaceTrack() = delete;

    explicit RaceTrack(const std::string &track_csv_path);

    // Given a 2d-coordinate, finds the nearest track-center coordinate index
    size_t findNearestTrackIndexBruteForce(const Vec2d &query_pt) const;

  private:
    std::vector<std::string> getCSVFilesInDirectory(const std::string &directory_path);

    // Calculates the gradient of a given point, as (h[x+1] - h[x-1])/2
    std::vector<float> gradient(const std::vector<float> &input_vec);

    std::string getTrackName(const std::string &filename);

    // Extracts the track data from the original csv files that are obtained from
    // https://github.com/TUMFTM/racetrack-database
    void getTrackDataFromCsv(const std::string &filename, TrackData &data_points_out);

    // Calculates the 2D extent of a track, given the x and y coordinates of the track
    Extent2d calculateTrackExtents(const TrackData &track_data_points);

    // Given the original track points and track extents shifts and if needed scales the track coordinates,
    // such that the entire track can fit into given window size.
    void centerTrackPointsToWindow(const Extent2d &track_extent,
                                   const float     window_width,
                                   const float     window_height,
                                   TrackData      &track_data_points_inout);

    void updateExtent(const std::vector<Vec2d> &vec, Extent2d &extent);

    // Given the track center coordinates and left & right lane widths, calculates the track boundaries on left and right
    void calculateTrackLanes(const TrackData    &track_data_points,
                             std::vector<Vec2d> &left_bound_inner,
                             std::vector<Vec2d> &left_bound_outer,
                             std::vector<Vec2d> &right_bound_inner,
                             std::vector<Vec2d> &right_bound_outer,
                             std::vector<float> &headings);

  public:
    std::string track_name_{};
    TrackData   track_data_points_{};
    //  data used for drawing
    std::vector<Vec2d> left_bound_inner_, left_bound_outer_, right_bound_inner_, right_bound_outer_;
    std::vector<Vec2d> start_line_, finish_line_;
    std::vector<float> headings_{};
};