#pragma once

#include <array>

namespace okitch
{

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
};

template <size_t WIDTH, size_t HEIGHT>
struct ImageMsg
{
    size_t                                  idx;
    std::array<uint8_t, WIDTH * HEIGHT * 4> data; //RGBA
};

template <size_t CAPACITY>
struct Laser2dMsg
{
    size_t                      size;
    std::array<Vec2d, CAPACITY> data;
};
} // namespace okitch