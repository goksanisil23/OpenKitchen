#pragma once

#include <cmath>
#include <math.h>
#include <vector>

constexpr int kScreenWidth  = 1600;
constexpr int kScreenHeight = 1400;

constexpr float kDeg2Rad{M_PI / 180.0F};

constexpr int kLeftBarrierColor[4]{255, 0, 0, 255};
constexpr int kRightBarrierColor[4]{0, 0, 255, 255};

struct Pixel
{
    int x{};
    int y{};
};

struct Vec2d
{
    float x{};
    float y{};

    float norm() const
    {
        return std::sqrt(x * x + y * y);
    }

    float squaredNorm() const
    {
        return x * x + y * y;
    }

    float distanceSquared(const Vec2d &other) const
    {
        return (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
    }

    Vec2d operator+(const Vec2d &other) const
    {
        return {x + other.x, y + other.y};
    }

    Vec2d operator/(const float scalar) const
    {
        return {x / scalar, y / scalar};
    }
};

struct Extent2d
{
    float min_x;
    float min_y;
    float max_x;
    float max_y;

    bool isPointInside(const Vec2d &pt)
    {
        if ((pt.x > min_x) && (pt.y > min_y) && (pt.x < max_x) && (pt.y < max_y))
            return true;
        return false;
    }
};

struct Ray_
{
    float x; // starting point of the ray
    float y;
    float angle; // global angle of the ray w.r.t screen [rad]
    float hit_x;
    float hit_y;
    bool  active{true}; // for crashed agents, used for early return in collision checker
};
