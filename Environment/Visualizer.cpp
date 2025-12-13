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

Visualizer::Visualizer(const bool hidden_window)
{
    SetTraceLogLevel(LOG_ERROR);

    if (hidden_window)
    {
        raylib::Window::SetConfigFlags(FLAG_WINDOW_HIDDEN);
    }
    window_ = std::make_unique<raylib::Window>(kScreenWidth, kScreenHeight, "");
    window_->SetPosition(40, 20);

    // We draw in this texture manner since we want to get pixel values
    render_target_ = LoadRenderTexture(kScreenWidth, kScreenHeight);
    view_rt_       = LoadRenderTexture(kScreenWidth, kScreenHeight);

    camera_.offset =
        Vector2{window_->GetWidth() / 2.0f, window_->GetHeight() / 2.0f}; // Center the camera view in the window
    camera_.rotation = 0.0f;
    camera_.zoom     = 3.0f;
}

void Visualizer::activateDrawing(bool const clear_background)
{
    render_target_.BeginMode();
    if (clear_background)
        window_->ClearBackground(BLACK);
}

void Visualizer::drawAgent(Agent &agent)
{
    raylib::Color color = (agent.crashed_)
                              ? raylib::Color((Color){253, 249, 0, 150})
                              : raylib::Color(agent.color_[0], agent.color_[1], agent.color_[2], agent.color_[3]);
    DrawCircle(agent.pos_.x, agent.pos_.y, agent.radius_, color);

    // Draw heading line for robot
    if (agent.draw_agent_heading_)
    {
        Vec2d       heading_end        = {agent.pos_.x + agent.radius_ * cos(kDeg2Rad * agent.rot_),
                                          agent.pos_.y + agent.radius_ * sin(kDeg2Rad * agent.rot_)};
        Vec2d const heading_circle_pos = (Vec2d{agent.pos_.x, agent.pos_.y} + heading_end) / 2.F;
        DrawCircle(heading_circle_pos.x, heading_circle_pos.y, agent.radius_ / 2.F, WHITE);
    }
}

void Visualizer::setAgentToFollow(const Agent *agent)
{
    agent_to_follow_ = agent;
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
        camera_.target   = {agent_to_follow_->pos_.x, agent_to_follow_->pos_.y};
        camera_.rotation = agent_to_follow_->rot_;

        // Use a separate view buffer to save the image rendered from the camera view
        // TODO: Remove this when image saving is not needed
        {
            view_rt_.BeginMode();
            window_->ClearBackground(BLACK);
            BeginMode2D(camera_);
            render_target_.GetTexture().Draw(
                {0.F, 0.F, (float)render_target_.texture.width, (float)-render_target_.texture.height},
                {0.F, 0.F},
                WHITE);
            EndMode2D();
            view_rt_.EndMode();
        }

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