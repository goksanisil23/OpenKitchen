#include "raylib-cpp.hpp"

#include "Npc.h"

// Non-playable-character

Npc::Npc(raylib::Vector2 pos, raylib::Vector2 vel, const bool is_goal) : position(pos), velocity(vel), is_goal(is_goal)
{
}

void Npc::move(float dt)
{
    // Move the Npc
    position = position + velocity * dt;

    // Check for collision with walls and bounce
    if (position.x < kRadius || position.x > GetScreenWidth() - kRadius)
    {
        velocity.x *= -1;
    }
    if (position.y < kRadius || position.y > GetScreenHeight() - kRadius)
    {
        velocity.y *= -1;
    }
}

void Npc::draw() const
{
    if (is_goal)
    {
        DrawCircleV(position, kRadius, DARKBLUE);
        DrawCircleLines(position.x, position.y, kRadius, BLUE);
    }
    else
    {
        DrawCircleV(position, kRadius, RED);
        DrawCircleLines(position.x, position.y, kRadius, DARKBROWN);
    }
}
