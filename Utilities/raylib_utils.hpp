#pragma once

#include "Driver.hpp"
#include "race_track_utils.hpp"
#include "raylib_msgs.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>

#include "raylib-cpp.hpp"

namespace okitch
{
const int kScreenWidth  = 1600;
const int kScreenHeight = 1400;

const raylib::Color kDrivableAreaCol{0, 255, 0, 255};
const raylib::Color kLeftBarrierCol{255, 0, 0, 255};
const raylib::Color kRightBarrierCol{0, 0, 255, 255};

inline bool isDrivableAreaPixel(const raylib::Color &pixel_color)
{
    if (pixel_color == kDrivableAreaCol)
        return true;
    return false;
}

inline bool isBarrierPixel(const raylib::Color &pixel_color)
{
    if ((pixel_color == kLeftBarrierCol) || (pixel_color == kRightBarrierCol))
        return true;
    return false;
}

// Visualization boilerplate class
class Vis
{
  public:
    Vis()
    {
        SetTraceLogLevel(LOG_ERROR);

        window_ = std::make_unique<raylib::Window>(kScreenWidth, kScreenHeight, "");
        window_->SetPosition(50, 20);
        window_->SetTargetFPS(60);

        // We draw in this texture manner since we want to get pixel values
        render_target_ = LoadRenderTexture(kScreenWidth, kScreenHeight);
    }

    void followDriver()
    {
        is_following_driver_ = true;

        camera_ = std::make_unique<raylib::Camera2D>();
        camera_->SetZoom(5.F);
        camera_->SetOffset({kScreenWidth / 2, kScreenHeight / 2});
        camera_->SetRotation(0);
    }

    void setCameraTarget(const Driver *followed_driver)
    {
        if (is_following_driver_)
        {
            followed_driver_ = followed_driver;
        }
    }

    void activateDrawing()
    {
        render_target_.BeginMode();
        // Update camera target position to follow the player
        if (is_following_driver_)
        {
            camera_->SetTarget(followed_driver_->pos_);
            camera_->BeginMode();
        }
        window_->ClearBackground(BLACK);
    }

    void drawDriver(Driver &driver, raylib::Image &render_buffer)
    {
        raylib::Color   driver_color = (driver.crashed_) ? raylib::Color::Yellow() : raylib::Color::DarkGray();
        raylib::Vector2 driver_texture_coord{driver.pos_.x, kScreenHeight - driver.pos_.y};
        render_buffer.DrawCircle(driver_texture_coord, driver.radius_, driver_color);
        // render_buffer.DrawText(
        //     std::to_string(driver.id_), {driver_texture_coord.x - 15 * driver.id_, driver_texture_coord.y}, 14, BLACK);

        // // Draw heading line for robot
        raylib::Vector2 heading_end = {driver_texture_coord.x + driver.radius_ * cos(DEG2RAD * driver.rot_),
                                       driver_texture_coord.y + driver.radius_ * -sin(DEG2RAD * driver.rot_)};
        render_buffer.DrawLine(driver_texture_coord, heading_end, WHITE);
    }

    // Check pixels that would be occupied by the outer edge of the robot, whether they instersect track boundaries
    bool checkDriverCollision(const raylib::Image &render_buffer, const Driver &driver)
    {
        raylib::Vector2 driver_texture_coord{driver.pos_.x, kScreenHeight - driver.pos_.y};
        raylib::Color  *pix_color =
            &((raylib::Color *)render_buffer.data)[static_cast<int>(driver_texture_coord.y) * render_buffer.width +
                                                   static_cast<int>(driver_texture_coord.x)];
        // std::cout << (unsigned)pix_color->r << " " << (unsigned)pix_color->g << " " << (unsigned)pix_color->b << " "
        //           << std::endl;

        // if (isDrivableAreaPixel(*pix_color))
        if (isBarrierPixel(*pix_color))
        {
            return true;
        }
        return false;
    }

    void enableImageSaving(const std::string &image_save_dir)
    {
        enable_img_saving_ = true;

        image_save_dir_ = image_save_dir;

        // Create directory for the training images to be saved
        auto save_dir_fs{std::filesystem::path(image_save_dir_)};
        if (!std::filesystem::exists(save_dir_fs))
        {
            std::filesystem::create_directory(save_dir_fs);
        }
    }

    void enableImageSharing(const char *shm_filename)
    {
        enable_img_sharing_ = true;

        // constexpr int kShmQueueSize{4};
        // q_ = shmmap<ImageMsg<kScreenWidth, kScreenHeight>, kShmQueueSize>(shm_filename);
        // assert(q_);
    }

    void disableDrawing()
    {
        static size_t ctr{0};

        if (is_following_driver_)
        {
            camera_->EndMode();
        }
        render_target_.EndMode();
        ctr++;
        if (enable_img_saving_)
        {
            if (ctr % img_saving_period_ == 0)
                saveImage();
        }
        if (enable_img_sharing_)
        {
            shareCurrentImage();
        }
    }

    void render()
    {
        window_->BeginDrawing();
        ClearBackground(BLACK);
        render_target_.GetTexture().Draw({0.F,
                                          0.F,
                                          static_cast<float>(render_target_.texture.width),
                                          static_cast<float>(-render_target_.texture.height)},
                                         {0.F, 0.F},
                                         WHITE);
        window_->DrawFPS();
        window_->EndDrawing();
    }

    void close()
    {
        // UnloadRenderTexture(render_target_);
        window_->Close();
    }

    void shareCurrentImage()
    {
        static size_t ctr{0};
        // raylib::Image img = LoadImageFromScreen();
        // Copy the image data to struct
        // raylib image is row-major order, RGBA/RGBA/RGBA/...
        // unsigned char *img_data = (unsigned char *)(img.data);
        // q_->write(
        //     [img_data](ImageMsg<kScreenWidth, kScreenHeight> &msg)
        //     {
        //         msg.idx = ctr;
        //         for (int i{0}; i < kScreenHeight * kScreenHeight * 4; i++)
        //         {
        //             msg.data[i] = (uint8_t)(img_data[i]);
        //         }
        //     });

        // Raylib-cpp's raylib::Image has a destructor that calls Unload() that free's the img.data
        // free(img.data);

        ctr++;
    }

    void saveImage()
    {
        static int32_t     img_ctr{0};
        std::ostringstream ss;
        ss << std::setfill('0') << std::setw(5) << img_ctr << ".png";
        std::string image_path = image_save_dir_ + ss.str();
        TakeScreenshot(image_path.c_str());
        Image screenshot = LoadImage(image_path.c_str());
        ExportImage(screenshot, (image_path).c_str());
        UnloadImage(screenshot);
        img_ctr++;
    }

  public:
    std::unique_ptr<raylib::Camera2D> camera_;
    std::unique_ptr<raylib::Window>   window_;
    raylib::RenderTexture             render_target_;
    raylib::Image                     current_frame_;

    Driver const *followed_driver_{nullptr};

    bool is_following_driver_{false};
    bool enable_img_saving_{false};
    bool enable_img_sharing_{false};

    const size_t img_saving_period_{1};
    std::string  image_save_dir_{};

    // Q<ImageMsg<kScreenWidth, kScreenHeight>, 4> *q_; // shared memory object
};

void drawArrow(raylib::Vector2 start, raylib::Vector2 end)
{
    raylib::Vector2 direction       = end - start;
    raylib::Vector2 normalized_dir  = direction.Normalize();
    constexpr float arrow_head_size = 20.0f;

    // Calculate the position of the arrowhead
    raylib::Vector2 arrow_head_pos = end;

    // Draw the arrow body (line)
    DrawLineV(start, end, raylib::Color::Yellow());

    // Draw the arrowhead (triangle)
    DrawTriangle(arrow_head_pos,
                 arrow_head_pos + (-normalized_dir.Rotate(45)) * arrow_head_size,
                 arrow_head_pos + (-normalized_dir.Rotate(-45)) * arrow_head_size,
                 raylib::Color::Yellow());
}

// Cross product of 2d raylib vectors
float crossProduct(raylib::Vector2 v1, raylib::Vector2 v2)
{
    return v1.x * v2.y - v1.y * v2.x;
}

void shadeAreaBetweenCurves(const std::vector<okitch::Vec2d> &curve_1,
                            const std::vector<okitch::Vec2d> &curve_2,
                            const raylib::Color               color)
{
    // We're shading 2 triangles that make up a trapezoid per iteration here.
    for (size_t i = 0; i < curve_1.size() - 1; ++i)
    {
        raylib::Vector2 v1 = {curve_1[i].x, curve_1[i].y};
        raylib::Vector2 v2 = {curve_2[i].x, curve_2[i].y};
        raylib::Vector2 v3 = {curve_1[i + 1].x, curve_1[i + 1].y};
        raylib::Vector2 v4 = {curve_2[i + 1].x, curve_2[i + 1].y};

        raylib::Vector2 side1 = {v2.x - v1.x, v2.y - v1.y};
        raylib::Vector2 side2 = {v3.x - v1.x, v3.y - v1.y};

        float cross = crossProduct(side1, side2);
        // If the cross product is negative, the vertices are already in counter-clockwise order.
        if (cross >= 0)
        {
            DrawTriangle(v1, v3, v2, color);
        }
        else
        {
            DrawTriangle(v1, v2, v3, color);
        }

        // other triangle
        side1 = raylib::Vector2{v3.x - v2.x, v3.y - v2.y};
        side2 = raylib::Vector2{v4.x - v2.x, v4.y - v2.y};
        cross = crossProduct(side1, side2);
        if (cross >= 0)
        {
            DrawTriangle(v2, v4, v3, color);
        }
        else
        {
            DrawTriangle(v2, v3, v4, color);
        }
    }
}

void drawLoadBar(const int32_t start_x,
                 const int32_t start_y,
                 const float   acceleration_delta,
                 const int32_t bar_length,
                 const bool    active)
{
    DrawRectangle(start_x, start_y, bar_length, 12, LIGHTGRAY);
    if (active)
    {
        if (acceleration_delta > 0.F)
        {
            DrawRectangle(start_x + bar_length / 2.f, start_y, (bar_length / 3.f), 12, BLUE);
        }
        else
        {
            DrawRectangle(start_x + bar_length / 2.f - (bar_length / 3.f), start_y, (bar_length / 3.f), 12, RED);
        }
    }

    DrawRectangleLines(start_x, start_y, bar_length, 12, GRAY);
}

} // namespace okitch