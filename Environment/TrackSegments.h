#pragma once

#include <vector>

#include "RaceTrack.h"
#include "Typedefs.h"

class TrackSegments
{
  public:
    explicit TrackSegments(const RaceTrack &race_track);
    ~TrackSegments();

    // Non-copyable
    TrackSegments(const TrackSegments &)            = delete;
    TrackSegments &operator=(const TrackSegments &) = delete;

    const Segment2d *getDeviceSegments() const
    {
        return d_segments_;
    }

    size_t getNumSegments() const
    {
        return num_segments_;
    }

  private:
    void addPolylineSegments(const std::vector<Vec2d> &polyline, std::vector<Segment2d> &segments);
    void uploadToDevice(const std::vector<Segment2d> &segments);

  private:
    Segment2d *d_segments_{nullptr}; // Device segment array
    size_t     num_segments_{0};     // Total segment count
};
