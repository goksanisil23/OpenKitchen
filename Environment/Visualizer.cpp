#include "Visualizer.h"

#include <iostream>
namespace env
{
namespace
{
float crossProduct(raylib::Vector2 v1, raylib::Vector2 v2)
{
    return v1.x * v2.y - v1.y * v2.x;
}
} // namespace

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

Visualizer::Visualizer()
{
    SetTraceLogLevel(LOG_ERROR);

    window_ = std::make_unique<raylib::Window>(kScreenWidth, kScreenHeight, "");
    window_->SetPosition(40, 20);

    // We draw in this texture manner since we want to get pixel values
    render_target_ = LoadRenderTexture(kScreenWidth, kScreenHeight);

    camera_.offset =
        Vector2{window_->GetWidth() / 2.0f, window_->GetHeight() / 2.0f}; // Center the camera view in the window
    camera_.rotation = 0.0f;
    camera_.zoom     = 3.0f;
}

void Visualizer::activateDrawing()
{
    render_target_.BeginMode();
    window_->ClearBackground(BLACK);
}

void Visualizer::drawAgent(Agent &agent, raylib::Image &render_buffer)
{
    raylib::Vector2 agent_texture_coord{agent.pos_.x, kScreenHeight - agent.pos_.y};

    raylib::Color color = (agent.crashed_) ? raylib::Color((Color){253, 249, 0, 150}) : agent.color_;
    render_buffer.DrawCircle(agent_texture_coord, agent.radius_, color);

    // Draw heading line for robot
    if (agent.draw_agent_heading_)
    {
        raylib::Vector2 heading_end = {agent_texture_coord.x + agent.radius_ * cos(DEG2RAD * agent.rot_),
                                       agent_texture_coord.y + agent.radius_ * -sin(DEG2RAD * agent.rot_)};
        // render_buffer.DrawLine(agent_texture_coord, heading_end, WHITE);
        render_buffer.DrawCircle((agent_texture_coord + heading_end) / 2.F, agent.radius_ / 2.F, WHITE);
    }

    if (agent.draw_sensor_rays_ && agent.has_raycast_sensor_)
    {
        drawSensorRays(agent.pixels_until_hit_, render_buffer);
    }
}

void Visualizer::setAgentToFollow(const Agent *agent)
{
    agent_to_follow_ = agent;
}

void Visualizer::updateSensor(Agent &agent, const raylib::Image &render_buffer)
{
    float ray_start_x, ray_start_y;
    float drv_rot_rad = agent.rot_ * DEG2RAD; // Convert angle to radians

    ray_start_x = agent.pos_.x + agent.sensor_offset_ * cos(DEG2RAD * agent.rot_);
    ray_start_y = kScreenHeight - agent.pos_.y + agent.sensor_offset_ * -sin(DEG2RAD * agent.rot_);

    if (agent.has_raycast_sensor_)
    {
        agent.sensor_hits_.clear();
        agent.pixels_until_hit_.clear();
        for (const auto angle : agent.sensor_ray_angles_)
        {
            const float end_pos_x = ray_start_x + agent.sensor_range_ * cos(DEG2RAD * (agent.rot_ + angle));
            const float end_pos_y = ray_start_y + agent.sensor_range_ * -sin(DEG2RAD * (agent.rot_ + angle));

            auto pixels_along_ray = bresenham(ray_start_x, ray_start_y, end_pos_x, end_pos_y);
            bool hit_found{false};
            for (const auto pix : pixels_along_ray)
            {
                // We check for the sensor hit based on the boundaries which are determined by the color
                const int   pix_idx{pix.y * render_buffer.width + pix.x};
                const Color pix_color = reinterpret_cast<Color *>(render_buffer.data)[pix_idx];
                if (Visualizer::isBarrierPixel(pix_color))
                {
                    // Calculate hit point relative to the robot
                    Vec2d hit_pt_relative;
                    float xTranslated = pix.x - ray_start_x;
                    float yTranslated = pix.y - ray_start_y;
                    hit_pt_relative.x = xTranslated * cos(drv_rot_rad) - yTranslated * sin(drv_rot_rad);
                    hit_pt_relative.y = xTranslated * sin(drv_rot_rad) + yTranslated * cos(drv_rot_rad);
                    agent.sensor_hits_.push_back(hit_pt_relative);
                    hit_found = true;
                    break;
                }
                else if (agent.draw_sensor_rays_) // drawing ray paths
                {
                    agent.pixels_until_hit_.push_back(pix);
                }
            }
            if (!hit_found) // If no hit is found, instead of no return, we report the max range
            {
                Vec2d max_range_pt_relative;
                float xTranslated       = end_pos_x - ray_start_x;
                float yTranslated       = end_pos_y - ray_start_y;
                max_range_pt_relative.x = xTranslated * cos(drv_rot_rad) - yTranslated * sin(drv_rot_rad);
                max_range_pt_relative.y = xTranslated * sin(drv_rot_rad) + yTranslated * cos(drv_rot_rad);
                agent.sensor_hits_.push_back(max_range_pt_relative);
            }
        }
    }
}

bool Visualizer::checkAgentCollision(const raylib::Image &render_buffer, const Agent &agent)
{
    raylib::Vector2      agent_texture_coord{agent.pos_.x, kScreenHeight - agent.pos_.y};
    const raylib::Color *pix_color =
        &((raylib::Color *)render_buffer.data)[static_cast<int>(agent_texture_coord.y) * render_buffer.width +
                                               static_cast<int>(agent_texture_coord.x)];

    if (Visualizer::isBarrierPixel(*pix_color))
    {
        return true;
    }
    return false;
}

void Visualizer::drawSensorRays(std::vector<Pixel> &pixels_until_hit, raylib::Image &render_buffer)
{

    for (const auto &pix : pixels_until_hit)
    {
        render_buffer.DrawPixel(pix.x, pix.y, raylib::Color::DarkPurple());
    }
    pixels_until_hit.clear();
}

void Visualizer::drawTrackTitle(const std::string &track_name)
{
    static Font custom_font = LoadFont("../Utils/CooperHewitt-Semibold.otf");
    DrawTextEx(custom_font, track_name.c_str(), {kScreenWidth / 2, 10}, 25, 3, ORANGE);
}

void Visualizer::render()
{

    if (!agent_to_follow_)
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
        if (user_draw_callback_)
            user_draw_callback_();
        window_->EndDrawing();
    }
    else
    {
        camera_.target = {agent_to_follow_->pos_.x, agent_to_follow_->pos_.y};
        window_->BeginDrawing();
        ClearBackground(BLACK);
        BeginMode2D(camera_);
        render_target_.GetTexture().Draw({0.F,
                                          0.F,
                                          static_cast<float>(render_target_.texture.width),
                                          static_cast<float>(-render_target_.texture.height)},
                                         {0.F, 0.F},
                                         WHITE);

        EndMode2D();
        window_->DrawFPS();
        if (user_draw_callback_)
            user_draw_callback_();
        window_->EndDrawing();
    }
}

void Visualizer::disableDrawing()
{
    render_target_.EndMode();
}

void Visualizer::close()
{
    // UnloadRenderTexture(render_target_);
    window_->Close();
}

void Visualizer::saveImage(const std::string &name)
{
    TakeScreenshot(name.c_str());
}

bool Visualizer::isDrivableAreaPixel(const raylib::Color &pixel_color)
{
    if (pixel_color == kDrivableAreaCol)
    {
        return true;
    }
    return false;
}

bool Visualizer::isBarrierPixel(const raylib::Color &pixel_color)
{
    if ((pixel_color == kLeftBarrierCol) || (pixel_color == kRightBarrierCol))
    {
        return true;
    }
    return false;
}

void Visualizer::shadeAreaBetweenCurves(const std::vector<Vec2d> &curve_1,
                                        const std::vector<Vec2d> &curve_2,
                                        const raylib::Color       color)
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

void Visualizer::drawArrow(raylib::Vector2 start, raylib::Vector2 end)
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

} // namespace env