#pragma once

#include "raylib-cpp.hpp"
#include <memory>

namespace okitch
{
// Visualization boilerplate class
class Vis
{
  public:
    Vis(const float window_width, const float window_height)
    {
        window_ = std::make_unique<raylib::Window>(window_width, window_height, "");
        window_->SetPosition(20, 20);
        window_->SetTargetFPS(60);

        // camera_ = std::make_unique<raylib::Camera2D>();
        // camera_->SetZoom(1.F);
    }

    // void checkPanAndZoom()
    // {
    //     // translate based on right click
    //     if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
    //     {
    //         Vector2 delta = GetMouseDelta();
    //         delta         = Vector2Scale(delta, -1.0f / camera_->zoom);

    //         camera_->target = Vector2Add(camera_->target, delta);
    //     }
    //     float wheel = GetMouseWheelMove();
    //     if (wheel != 0)
    //     {
    //         // get the world point that is under the mouse
    //         Vector2 mouseWorldPos = GetScreenToWorld2D(GetMousePosition(), *camera_);
    //         camera_->offset       = GetMousePosition();
    //         camera_->target       = mouseWorldPos;
    //         camera_->zoom += wheel * 0.125f;
    //         if (camera_->zoom < 0.125f)
    //             camera_->zoom = 0.125f;
    //     }
    // }

    void activateDrawing()
    {
        // checkPanAndZoom();

        window_->BeginDrawing();
        window_->ClearBackground(BLACK);
        // camera_->BeginMode();
    }

    void deactivateDrawing()
    {
        // camera_->EndMode();
        window_->DrawFPS();
        window_->EndDrawing();
    }

  public:
    // std::unique_ptr<raylib::Camera2D> camera_;
    std::unique_ptr<raylib::Window> window_;
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

raylib::Vector2 *convertToRaylibArray(const std::vector<float> &x, const std::vector<float> &y)
{
    raylib::Vector2 *raylib_arr = new raylib::Vector2[x.size()];
    for (size_t i{}; i < x.size(); i++)
    {
        raylib_arr[i].x = x[i];
        raylib_arr[i].y = y[i];
    }

    return raylib_arr;
}

// Comparison function to be used with std::qsort, hence the specific signature
int compareVector2(const void *a, const void *b)
{
    const raylib::Vector2 *v1 = static_cast<const raylib::Vector2 *>(a);
    const raylib::Vector2 *v2 = static_cast<const raylib::Vector2 *>(b);

    if (v1->x < v2->x)
    {
        return -1;
    }
    if (v1->x > v2->x)
    {
        return 1;
    }
    return 0;
}

// Cross product of 2d raylib vectors
float crossProduct(raylib::Vector2 v1, raylib::Vector2 v2)
{
    return v1.x * v2.y - v1.y * v2.x;
}

void shadeAreaBetweenCurves(raylib::Vector2    *curve_1,
                            raylib::Vector2    *curve_2,
                            const size_t        size,
                            const raylib::Color color)
{
    // We're shading 2 triangles that make up a trapezoid per iteration here.
    for (size_t i = 0; i < size - 1; ++i)
    {
        raylib::Vector2 v1 = curve_1[i];
        raylib::Vector2 v2 = curve_2[i];
        raylib::Vector2 v3 = curve_1[i + 1];
        raylib::Vector2 v4 = curve_2[i + 1];

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