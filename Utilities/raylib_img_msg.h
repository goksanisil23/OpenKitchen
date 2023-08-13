#pragma once

#include <array>

namespace okitch
{

template <size_t WIDTH, size_t HEIGHT>
struct SharedMsg
{
    size_t                                  idx;
    std::array<uint8_t, WIDTH * HEIGHT * 4> data; //RGBA
};
} // namespace okitch