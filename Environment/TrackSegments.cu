#include <cuda_runtime.h>

#include "TrackSegments.h"
#include "Typedefs.h"

TrackSegments::TrackSegments(const RaceTrack &race_track)
{
    std::vector<Segment2d> segments;

    // Convert all boundary polylines to segments
    addPolylineSegments(race_track.left_bound_inner_, segments);
    addPolylineSegments(race_track.left_bound_outer_, segments);
    addPolylineSegments(race_track.right_bound_inner_, segments);
    addPolylineSegments(race_track.right_bound_outer_, segments);

    // Add continuous loop segments (connecting start to finish)
    if (race_track.left_bound_inner_.size() > 1)
    {
        // Connect last point to first point for inner boundaries
        const auto &li_front = race_track.left_bound_inner_.front();
        const auto &li_back  = race_track.left_bound_inner_.back();
        segments.push_back({li_back.x, li_back.y, li_front.x, li_front.y});

        const auto &ri_front = race_track.right_bound_inner_.front();
        const auto &ri_back  = race_track.right_bound_inner_.back();
        segments.push_back({ri_back.x, ri_back.y, ri_front.x, ri_front.y});
    }

    if (race_track.left_bound_outer_.size() > 1)
    {
        // Connect last point to first point for outer boundaries
        const auto &lo_front = race_track.left_bound_outer_.front();
        const auto &lo_back  = race_track.left_bound_outer_.back();
        segments.push_back({lo_back.x, lo_back.y, lo_front.x, lo_front.y});

        const auto &ro_front = race_track.right_bound_outer_.front();
        const auto &ro_back  = race_track.right_bound_outer_.back();
        segments.push_back({ro_back.x, ro_back.y, ro_front.x, ro_front.y});
    }

    uploadToDevice(segments);
}

TrackSegments::~TrackSegments()
{
    if (d_segments_)
    {
        cudaFree(d_segments_);
        d_segments_ = nullptr;
    }
}

void TrackSegments::addPolylineSegments(const std::vector<Vec2d> &polyline, std::vector<Segment2d> &segments)
{
    if (polyline.size() < 2)
        return;

    for (size_t i = 0; i < polyline.size() - 1; ++i)
    {
        Segment2d seg;
        seg.x1 = polyline[i].x;
        seg.y1 = polyline[i].y;
        seg.x2 = polyline[i + 1].x;
        seg.y2 = polyline[i + 1].y;
        segments.push_back(seg);
    }
}

void TrackSegments::uploadToDevice(const std::vector<Segment2d> &segments)
{
    num_segments_ = segments.size();
    GOX_ASSERT(num_segments_ > 0);

    cudaMalloc(&d_segments_, num_segments_ * sizeof(Segment2d));
    cudaMemcpy(d_segments_, segments.data(), num_segments_ * sizeof(Segment2d), cudaMemcpyHostToDevice);
}
