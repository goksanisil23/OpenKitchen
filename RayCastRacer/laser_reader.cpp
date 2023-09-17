#include <cassert>
#include <chrono>
#include <thread>

#include "raylib-cpp.hpp"
#include "raylib_msgs.h"
#include "spmc_queue.h"

using namespace std;
constexpr int WIDTH  = 1000;
constexpr int HEIGHT = 1000;

constexpr size_t LASER_MSG_CAP  = 100;
constexpr size_t SHM_QUEUE_SIZE = 4;

// use taskset -c to bind core
int main(int argc, char **argv)
{

    const int screenWidth  = 800;
    const int screenHeight = 450;
    InitWindow(screenWidth, screenHeight, "raylib [core] example - 2D point cloud");

    auto sensor_msg_shm_q =
        shmmap<okitch::Laser2dMsg<LASER_MSG_CAP>, SHM_QUEUE_SIZE>("laser_msgs"); // shared memory object
    assert(sensor_msg_shm_q);

    auto reader = sensor_msg_shm_q->getReader();

    std::vector<okitch::Laser2dMsg<LASER_MSG_CAP>> sensor_hits;
    sensor_hits.reserve(LASER_MSG_CAP);
    while (!WindowShouldClose())
    {
        okitch::Laser2dMsg<LASER_MSG_CAP> *msg = reader.readLast();
        if (!msg)
        {
            // std::cout << "No msg yet..." << std::endl;
            continue;
        }
        // sensor_hits.clear();
        // for (size_t i{0}; i < msg->size; i++)
        // {
        //     sensor_hits.push_back(msg->data[i]);
        // }

        // Draw
        BeginDrawing();
        ClearBackground(RAYWHITE);
        for (size_t i = 0; i < msg->size; ++i)
        {
            std::cout << msg->data[i].norm() << " ";
            DrawCircleV({msg->data[i].x + screenWidth / 2.F, msg->data[i].y + screenHeight / 2.F}, 3, DARKGRAY);
        }
        std::cout << "\n--------------------" << std::endl;
        EndDrawing();
    }
    CloseWindow();

    return 0;
}
