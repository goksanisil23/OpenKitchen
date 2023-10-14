#pragma once

#include "Typedefs.h"
#include <array>

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
