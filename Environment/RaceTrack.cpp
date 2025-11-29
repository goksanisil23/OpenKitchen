#include "RaceTrack.h"

RaceTrack::RaceTrack(const std::string &track_csv_path) : track_name_{getTrackName(track_csv_path)}
{
    getTrackDataFromCsv(track_csv_path, track_data_points_);

    const auto track_extent = calculateTrackExtents(track_data_points_);
    centerTrackPointsToWindow(track_extent, kScreenWidth, kScreenHeight, track_data_points_);
    // Calculate track boundaries and lane centers
    calculateTrackLanes(
        track_data_points_, left_bound_inner_, left_bound_outer_, right_bound_inner_, right_bound_outer_, headings_);
    start_line_  = {right_bound_outer_.front(), right_bound_outer_.back()};
    finish_line_ = {left_bound_outer_.front(), left_bound_outer_.back()};
}

size_t RaceTrack::findNearestTrackIndexBruteForce(const Vec2d &query_pt) const
{
    float  min_distance = std::numeric_limits<float>::max();
    size_t min_idx{0};
    float  distance;
    for (size_t i{0}; i < track_data_points_.x_m.size(); i++)
    {
        distance = query_pt.distanceSquared({track_data_points_.x_m[i], track_data_points_.y_m[i]});
        if (distance < min_distance)
        {
            min_distance = distance;
            min_idx      = i;
        }
    }
    return min_idx;
}

float RaceTrack::getNearestDistanceToTrackBoundary(const Vec2d &query_pt) const
{
    float min_distance = std::numeric_limits<float>::max();
    float distance;
    for (size_t i{0}; i < left_bound_inner_.size(); i++)
    {
        distance = query_pt.distanceSquared(left_bound_inner_[i]);
        if (distance < min_distance)
        {
            min_distance = distance;
        }
        distance = query_pt.distanceSquared(right_bound_inner_[i]);
        if (distance < min_distance)
        {
            min_distance = distance;
        }
    }
    return std::sqrt(min_distance);
}

std::vector<std::string> RaceTrack::getCSVFilesInDirectory(const std::string &directory_path)
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

std::vector<float> RaceTrack::gradient(const std::vector<float> &input_vec)
{
    std::vector<float> out_grad_vec;
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
    return out_grad_vec;
}

std::string RaceTrack::getTrackName(const std::string &filename)
{
    size_t last_slash = filename.rfind('/');
    size_t last_dot   = filename.rfind('.');

    std::string track_name = filename.substr(last_slash + 1, last_dot - last_slash - 1);
    std::transform(
        track_name.begin(), track_name.end(), track_name.begin(), [](unsigned char c) { return std::toupper(c); });
    return track_name;
}

void RaceTrack::getTrackDataFromCsv(const std::string &filename, TrackData &data_points_out)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    std::string line;
    // Skip the header line
    std::getline(file, line);
    constexpr float kTrackWidthScale{3.0};
    constexpr float kTrackWidthMaxLimit{17.0};
    constexpr float kTrackWidthMinLimit{4.0};

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
        data_points_out.w_tr_right_m.push_back(
            std::min(std::max(kTrackWidthMinLimit, std::stof(token)) * kTrackWidthScale, kTrackWidthMaxLimit));

        std::getline(iss, token, ',');
        data_points_out.w_tr_left_m.push_back(
            std::min(std::max(kTrackWidthMinLimit, std::stof(token)) * kTrackWidthScale, kTrackWidthMaxLimit));
    }

    file.close();
}

Extent2d RaceTrack::calculateTrackExtents(const TrackData &track_data_points)
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

void RaceTrack::centerTrackPointsToWindow(const Extent2d &track_extent,
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

void RaceTrack::updateExtent(const std::vector<Vec2d> &vec, Extent2d &extent)
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

// Given the track center coordinates and left & right lane widths, calculates the track boundaries on left and right
void RaceTrack::calculateTrackLanes(const TrackData    &track_data_points,
                                    std::vector<Vec2d> &left_bound_inner,
                                    std::vector<Vec2d> &left_bound_outer,
                                    std::vector<Vec2d> &right_bound_inner,
                                    std::vector<Vec2d> &right_bound_outer,
                                    std::vector<float> &headings)
{
    // Find the heading of the line so that we can draw a perpendicular point, lane width away
    std::vector<float> dx = gradient(track_data_points.x_m);
    std::vector<float> dy = gradient(track_data_points.y_m);

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
        headings.push_back(std::atan2(dy[i], dx[i]) * 180.0F / M_PI);
    }

    left_bound_inner.resize(dx.size());
    left_bound_outer.resize(dx.size());
    right_bound_inner.resize(dx.size());
    right_bound_outer.resize(dx.size());

    constexpr float kBoundaryThickness{3.F};

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
