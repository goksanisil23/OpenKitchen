#include "Npc.h"
#include "Robot.h"

#include <iostream>
#include <unistd.h>

#include "raylib-cpp.hpp"

constexpr int kScreenWidth  = 1600;
constexpr int kScreenHeight = 1200;

constexpr int   kNumObstacles = 100;
constexpr float kDt{0.01};

bool reachedGoal(const Npc &goal, const Robot &robot)
{
    if ((goal.position - robot.state_.position).Length() < (Robot::kRadius + Npc::kRadius))
    {
        return true;
    }
    return false;
}

bool crashedToObstacle(std::vector<Npc> &obstacles, const Robot &robot)
{
    for (auto &obs : obstacles)
    {
        if ((obs.position - robot.state_.position).Length() < (Robot::kRadius + Npc::kRadius))
        {
            // Reset this obstacle
            obs.position = raylib::Vector2(GetRandomValue(0, kScreenWidth), GetRandomValue(0, kScreenHeight));
            obs.velocity = raylib::Vector2(GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit),
                                           GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit));
            return true;
        }
    }
    return false;
}

int main()
{

    raylib::Window window(kScreenWidth, kScreenHeight, "Differential Drive Robot Simulation");

    // Initialize obstacles
    std::vector<Npc> obstacles, obstacles_future;
    obstacles.reserve(kNumObstacles);
    obstacles_future.reserve(kNumObstacles);
    for (int i = 0; i < kNumObstacles; ++i)
    {
        obstacles.emplace_back(raylib::Vector2(GetRandomValue(0, kScreenWidth), GetRandomValue(0, kScreenHeight)),
                               raylib::Vector2(GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit),
                                               GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit)));
    }
    Npc goal(raylib::Vector2(GetRandomValue(0, kScreenWidth), GetRandomValue(0, kScreenHeight)),
             raylib::Vector2(GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit),
                             GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit)),
             true);
    Npc goal_future = goal;

    // Initialize robot
    Robot                     robot(raylib::Vector2(kScreenWidth / 2, kScreenHeight / 2), 0.0f);
    std::vector<Robot::State> dynamic_window_possible_states;

    SetTargetFPS(100);

    // Main game loop
    while (!window.ShouldClose())
    {
        // Copy the current state of obstacles for planning
        obstacles_future = obstacles;
        goal_future      = goal;
        for (int i{0}; i < Robot::kNumHorizonStep; i++)
        {
            for (auto &obs : obstacles_future)
            {
                obs.move(kDt);
            }
            goal_future.move(kDt);
        }

        // Use future obstacle states for planning
        auto chosen_action = robot.chooseAction(
            obstacles_future, goal_future, kDt * Robot::kNumHorizonStep, dynamic_window_possible_states);
        robot.v_wheels_ = chosen_action;

        for (auto &obstacle : obstacles)
        {
            obstacle.move(kDt);
        }
        goal.move(kDt);

        auto new_state = robot.iterateKinematics(chosen_action, robot.state_, kDt);
        robot.state_   = new_state;
        std::cout << "action: " << chosen_action.v_left << " " << chosen_action.v_right << std::endl;

        // Draw
        BeginDrawing();
        ClearBackground(BLACK);

        for (const auto &obstacle : obstacles)
        {
            obstacle.draw();
        }
        goal.draw();

        robot.draw();

        if (crashedToObstacle(obstacles, robot))
        {
            sleep(3);
            std::cerr << "Crashed!" << std::endl;
        }
        else if (reachedGoal(goal, robot))
        {
            // Reset goal
            goal.position = raylib::Vector2(GetRandomValue(0, kScreenWidth), GetRandomValue(0, kScreenHeight));
            goal.velocity = raylib::Vector2(GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit),
                                            GetRandomValue(-Npc::kVelocityLimit, Npc::kVelocityLimit));
        }

        EndDrawing();
    }

    return 0;
}
