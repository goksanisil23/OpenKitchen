#pragma once

#include "raylib-cpp.hpp"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>

#include "Agent.h"
#include "IpcMsgs.h"
#include "Typedefs.h"

namespace env
{
const raylib::Color kDrivableAreaCol{0, 255, 0, 255};
const raylib::Color kLeftBarrierCol{255, 0, 0, 255};
const raylib::Color kRightBarrierCol{0, 0, 255, 255};

constexpr bool kContinuousLoop{true};

// Visualization boilerplate class
class Visualizer
{
  public:
    Visualizer();

    void activateDrawing();

    void drawAgent(Agent &agent, raylib::Image &render_buffer);

    // returns sensonr hit points and pixels along the ray until the hit points
    void updateSensor(Agent &agent, const raylib::Image &render_buffer);

    void drawTrackTitle(const std::string &name);

    void drawSensorRays(std::vector<Pixel> &pixels_until_hit, raylib::Image &render_buffer);

    // Check pixels that would be occupied by the outer edge of the robot, whether they instersect track boundaries
    bool checkAgentCollision(const raylib::Image &render_buffer, const Agent &agent);

    void enableImageSaving(const std::string &image_save_dir);

    void enableImageSharing(const char *shm_filename);

    void disableDrawing();

    void render();

    void close();

    void shareCurrentImage();
    void saveImage();

    static bool isDrivableAreaPixel(const raylib::Color &pixel_color);
    static bool isBarrierPixel(const raylib::Color &pixel_color);

    static void drawArrow(raylib::Vector2 start, raylib::Vector2 end);

    static void shadeAreaBetweenCurves(const std::vector<Vec2d> &curve_1,
                                       const std::vector<Vec2d> &curve_2,
                                       const raylib::Color       color);

  public:
    std::unique_ptr<raylib::Camera2D> camera_;
    std::unique_ptr<raylib::Window>   window_;
    raylib::RenderTexture             render_target_;
    raylib::Image                     current_frame_;

    bool enable_img_saving_{false};
    bool enable_img_sharing_{false};

    const size_t img_saving_period_{1};
    std::string  image_save_dir_{};

    // Q<ImageMsg<kScreenWidth, kScreenHeight>, 4> *q_; // shared memory object
};

} // namespace env