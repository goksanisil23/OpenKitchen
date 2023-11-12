#pragma once

#include "raylib-cpp.hpp"

// Non-playable-character
class Npc
{
  public:
    static constexpr float kRadius{20.0f};
    static constexpr float kVelocityLimit{100.0F};

    raylib::Vector2 position;
    raylib::Vector2 velocity;

    bool is_goal{false};

    Npc(raylib::Vector2 pos, raylib::Vector2 vel, const bool is_goal = false);

    void move(float dt);

    void draw() const;
};