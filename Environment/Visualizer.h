#pragma once

#include "raylib-cpp.hpp"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <functional>

#include "Agent.h"
#include "IpcMsgs.h"
#include "Typedefs.h"

namespace env
{
const raylib::Color kDrivableAreaCol{0, 255, 0, 255};

constexpr bool kContinuousLoop{true};

// Visualization boilerplate class
class Visualizer
{
  public:
    Visualizer();

    void activateDrawing(bool const clear_background = true);

    void drawAgent(Agent &agent);

    // When called, the camera follows this agent
    void setAgentToFollow(const Agent *agent);

    void drawTrackTitle(const std::string &name);

    void disableDrawing();

    void render();

    void close();

    void saveImage(const std::string &name);

    static void drawArrow(raylib::Vector2 start, raylib::Vector2 end);

    static void shadeAreaBetweenCurves(const std::vector<Vec2d> &curve_1,
                                       const std::vector<Vec2d> &curve_2,
                                       const raylib::Color       color);

  public:
    raylib::Camera2D                camera_;
    std::unique_ptr<raylib::Window> window_;
    raylib::RenderTexture           render_target_;

    const Agent *agent_to_follow_{nullptr};

    std::function<void()> user_draw_callback_{nullptr};

    // Q<ImageMsg<kScreenWidth, kScreenHeight>, 4> *q_; // shared memory object
};

} // namespace env