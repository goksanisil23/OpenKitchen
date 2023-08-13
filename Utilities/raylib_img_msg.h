#pragma once

#include <array>

namespace okitch
{

const char *shm_file_semseg_in = "raylib_semseg_input_shmem";

template <size_t WIDTH, size_t HEIGHT>
struct SharedMsg
{
    size_t                                  idx;
    std::array<uint8_t, WIDTH * HEIGHT * 4> data; //RGBA
};
} // namespace okitch