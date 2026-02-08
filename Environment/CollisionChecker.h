#pragma once

#include <memory>

#include "Agent.h"
#include "Typedefs.h"

class CollisionChecker
{
  public:
    CollisionChecker(const Segment2d *d_segments, size_t num_segments, const std::vector<Agent *> &agents);
    ~CollisionChecker();

    // Returns true if a collision is detected.
    void checkCollision();

    // Get ray data for visualization (call after checkCollision)
    const Ray_ *getHostRays() const;
    size_t      getNumRays() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
