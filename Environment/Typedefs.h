#pragma once

#include <math.h>
#include <vector>

constexpr int kScreenWidth  = 1600;
constexpr int kScreenHeight = 1400;

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

    float distanceSquared(const Vec2d &other) const
    {
        return (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
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