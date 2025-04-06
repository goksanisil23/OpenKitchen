#pragma once

#include <memory>

#include "Agent.h"

class CollisionChecker
{
  public:
    CollisionChecker(const unsigned int framebuffer_obj_id, const std::vector<Agent *> agents);
    ~CollisionChecker();

    // Returns true if a collision is detected.
    void checkCollision();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
