#pragma once

#include <cstdint>

enum class HitType
{
    kObstacle,
    kEnemy,
    kUndefined
};

struct Ray_
{
    float   x; // starting point of the ray
    float   y;
    float   angle;    // global angle of the ray w.r.t screen
    float   hit_dist; // squared hit distance
    HitType hit_type; // type of the object hit by the ray
};

struct Agent
{
    float x, y, velocity, heading;
};

constexpr int32_t NUM_AGENTS_TEAM_A = 5;
constexpr int32_t NUM_AGENTS_TEAM_B = 8;

constexpr int32_t NUM_RAYS_PER_AGENT = 20;
constexpr int32_t NUM_AGENTS         = NUM_AGENTS_TEAM_A + NUM_AGENTS_TEAM_B;
constexpr int32_t NUM_TOTAL_RAYS     = NUM_RAYS_PER_AGENT * NUM_AGENTS;

constexpr int32_t SCREEN_WIDTH  = 1400;
constexpr int32_t SCREEN_HEIGHT = 1000;