#pragma once

#include <memory>

#include "Agent.h"

class CollisionChecker
{
  public:
    CollisionChecker(const unsigned int texture_id, const std::vector<Agent *> &agents, const bool draw_rays = true);
    ~CollisionChecker();

    // Returns true if a collision is detected.
    void checkCollision();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
