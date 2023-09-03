#pragma once

#include "race_track_utils.hpp"
#include "raylib_msgs.h"

#include <cassert>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>

#include "raylib-cpp.hpp"

namespace okitch
{
constexpr int kScreenWidth  = 1000;
constexpr int kScreenHeight = 1000;

namespace
{

std::vector<Pixel> bresenham(const int &x0, const int &y0, const int &x1, const int &y1)
{
    std::vector<Pixel> pixels;

    const auto dx = std::abs(x1 - x0);
    const auto dy = std::abs(y1 - y0);

    auto x = x0;
    auto y = y0;

    int sx = (x0 > x1) ? -1 : 1;
    int sy = (y0 > y1) ? -1 : 1;

    if (dx > dy)
    {
        auto err = dx / 2.0F;
        while (x != x1)
        {
            pixels.push_back({x, y});
            err -= dy;
            if (err < 0.)
            {
                y += sy;
                err += dx;
            }
            x += sx;
        }
    }
    else
    {
        auto err = dy / 2.0;
        while (y != y1)
        {
            pixels.push_back({x, y});
            err -= dx;
            if (err < 0)
            {
                x += sx;
                err += dy;
            }
            y += sy;
        }
    }

    pixels.push_back({x, y});

    return pixels;
}

} // namespace
class Driver
{
  public:
    Driver(raylib::Vector2 start_pos, float start_rot) : pos_{start_pos}, rot_{start_rot}
    {
    }

    void enableRaycastSensor()
    {
        has_raycast_sensor_ = true;
        for (int i = -90; i <= 90; i += 3)
        {
            sensor_ray_angles_.push_back(static_cast<float>(i));
        }
    }

    // Kinematics update of the driver
    void update()
    {
        constexpr float kLongSpeed = 100.0f;
        constexpr float kRotSpeed  = 5.0f;
        // Update
        if (IsKeyDown(KEY_RIGHT))
            rot_ += kRotSpeed;
        if (IsKeyDown(KEY_LEFT))
            rot_ -= kRotSpeed;

        if (IsKeyDown(KEY_UP))
        {
            pos_.x += cos(DEG2RAD * rot_) * kLongSpeed * GetFrameTime();
            pos_.y += sin(DEG2RAD * rot_) * kLongSpeed * GetFrameTime();
        }
        else if (IsKeyDown(KEY_DOWN))
        {
            pos_.x -= cos(DEG2RAD * rot_) * kLongSpeed * GetFrameTime();
            pos_.y -= sin(DEG2RAD * rot_) * kLongSpeed * GetFrameTime();
        }
    }

    void updateSensor(raylib::Image &render_buffer, std::vector<Vec2d> &hitpoints_out)
    {
        if (has_raycast_sensor_)
        {
            hitpoints_out.clear();
            for (const auto angle : sensor_ray_angles_)
            {
                const float end_pos_x = 500.F + sensor_range_ * 5.F * cos(DEG2RAD * (rot_ + angle));
                const float end_pos_y = 500.F + sensor_range_ * 5.F * -sin(DEG2RAD * (rot_ + angle));

                auto pixels_along_ray = okitch::bresenham(500.F, 500.F, end_pos_x, end_pos_y);
                for (const auto pix : pixels_along_ray)
                {
                    Color *pix_color = &((Color *)render_buffer.data)[pix.y * render_buffer.width + pix.x];
                    if (((pix_color->r == 255) && (pix_color->g == 0) && (pix_color->b == 0)) ||
                        ((pix_color->r == 0) && (pix_color->g == 0) && (pix_color->b == 255)))
                    {
                        // Show hitpoints as white
                        pix_color->r = 255;
                        pix_color->g = 255;
                        pix_color->b = 255;
                        // Calculate hit point relative to the robot
                        Vec2d hit_pt_relative;
                        float xTranslated    = pix.x - 500.F;
                        float yTranslated    = pix.y - 500.F;
                        float angleInRadians = rot_ * DEG2RAD; // Convert angle to radians
                        hit_pt_relative.x    = xTranslated * cos(angleInRadians) - yTranslated * sin(angleInRadians);
                        hit_pt_relative.y    = xTranslated * sin(angleInRadians) + yTranslated * cos(angleInRadians);
                        hitpoints_out.push_back(hit_pt_relative);
                        break;
                    }
                    else // drawing ray paths
                    {
                        pix_color->r = 100;
                        pix_color->g = 100;
                        pix_color->b = 100;
                    }
                }
            }
        }
    }

    raylib::Vector2 pos_;
    float           rot_;
    raylib::Vector2 size_{1.9, 4.572}; // width/height based on Porsche-911

    bool                has_raycast_sensor_{false};
    std::vector<float>  sensor_ray_angles_;
    const raylib::Color ray_color_ = raylib::Color::Magenta();
    const float         sensor_range_{80.0F};
};

// Visualization boilerplate class
class Vis
{
  public:
    Vis(const float window_width, const float window_height)
    {
        window_ = std::make_unique<raylib::Window>(window_width, window_height, "");
        window_->SetPosition(20, 20);
        window_->SetTargetFPS(60);

        camera_ = std::make_unique<raylib::Camera2D>();
        camera_->SetZoom(5.F);
        camera_->SetOffset({window_width / 2, window_height / 2});
        camera_->SetRotation(0);

        // We draw in this texture manner since we want to get pixel values
        render_target_ = LoadRenderTexture(window_width, window_height);
    }

    void activateDrawing(const Driver &driver)
    {
        // Update camera target position to follow the player
        camera_->SetTarget(driver.pos_);
        // Begin the drawing mode
        render_target_.BeginMode();
        camera_->BeginMode();
        window_->ClearBackground(BLACK);
    }

    void drawDriver(const Driver &driver)
    {
        DrawCircleV(driver.pos_, driver.size_.x, DARKGRAY);

        // Draw heading line for robot
        raylib::Vector2 heading_end = {driver.pos_.x + driver.size_.x * cos(DEG2RAD * driver.rot_),
                                       driver.pos_.y + driver.size_.x * sin(DEG2RAD * driver.rot_)};
        DrawLineEx(driver.pos_, heading_end, driver.size_.x / 2.F, WHITE);
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

        camera_->EndMode();
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
    bool                              enable_img_saving_{false};
    const size_t                      img_saving_period_{1};
    bool                              enable_img_sharing_{false};
    std::string                       image_save_dir_{};

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

void shadeAreaBetweenCurves(const std::vector<race_track_gen::Vec2> &curve_1,
                            const std::vector<race_track_gen::Vec2> &curve_2,
                            const raylib::Color                      color)
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

} // namespace okitch