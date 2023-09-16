#pragma once

#include "raylib-cpp.hpp"

namespace okitch
{

extern const int kScreenWidth;
extern const int kScreenHeight;

namespace
{

const raylib::Color kLeftBoudaryCol{255, 0, 0, 255};
const raylib::Color kLRightBoudaryCol{0, 0, 255, 255};
constexpr bool      kDrawSensorRays{false};

constexpr float    kDeltaStandstillLimit{0.00001F}; // displacement threshold used to indicate an agent is standstill
constexpr uint32_t kStandstillTimeout{200}; // # of consecutive standstill iterations after which we reset the episode

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

// Checks whether the pixel is a boundary pixel based on it's color
inline bool isBoundaryPixel(const raylib::Color &pixel_color)
{
    if ((pixel_color == kLeftBoudaryCol) || (pixel_color == kLRightBoudaryCol))
        return true;
    return false;
}

} // namespace

class Driver
{
  public:
    Driver(raylib::Vector2 start_pos, float start_rot, int16_t id, const bool is_cam_following)
        : pos_{start_pos}, rot_{start_rot}, id_{id}, is_cam_following_{is_cam_following}
    {
        if (is_cam_following)
        {
            sensor_range_ = 80.F;
        }
        else
        {
            sensor_range_ = 20.F;
        }
    }

    void enableRaycastSensor()
    {
        has_raycast_sensor_ = true;
        // Denser pattern towards the middle
        for (int i = -70; i <= 70; i += 1)
        {
            if (std::abs(i) < 15)
                sensor_ray_angles_.push_back(static_cast<float>(i));
            else if (std::abs(i) > 15 && i % 3 == 0)
                sensor_ray_angles_.push_back(static_cast<float>(i));
        }
    }

    void enableManualControl()
    {
        manual_control_enabled_ = true;
    }

    void enableAutoControl()
    {
        auto_control_enabled_ = true;
    }

    // Kinematics update of the driver, when controlled via keyboard by the user
    void updateManualControl()
    {
        if (manual_control_enabled_)
        {
            if (IsKeyDown(KEY_RIGHT))
            {
                rot_ += kRotSpeed;
            }
            else if (IsKeyDown(KEY_LEFT))
            {
                rot_ -= kRotSpeed;
            }

            if (IsKeyDown(KEY_UP))
            {
                speed_ += kAccInc;
            }
            else if (IsKeyDown(KEY_DOWN))
            {
                speed_ -= kAccInc;
            }

            pos_.x += cos(DEG2RAD * rot_) * speed_ * GetFrameTime();
            pos_.y += sin(DEG2RAD * rot_) * speed_ * GetFrameTime();
        }
    }

    void updateAutoControl(const float acceleration_delta, const float steering_delta)
    {
        constexpr float kSpeedLimit{100.F};

        if (auto_control_enabled_)
        {
            rot_ += steering_delta;

            acceleration_ += acceleration_delta;
            speed_ += (acceleration_ * GetFrameTime());

            speed_ = (speed_ < -kSpeedLimit) ? -kSpeedLimit : speed_;
            speed_ = (speed_ > kSpeedLimit) ? kSpeedLimit : speed_;
            // std::cout << "robot: " << id_ << " speed: " << speed_ << " rot: " << rot_ << std::endl;

            float delta_x = cos(DEG2RAD * rot_) * speed_ * GetFrameTime();
            pos_.x += delta_x;
            float delta_y = sin(DEG2RAD * rot_) * speed_ * GetFrameTime();
            pos_.y += delta_y;

            if ((std::abs(delta_x) < kDeltaStandstillLimit) && (std::abs(delta_y) < kDeltaStandstillLimit))
            {
                standstill_ctr_++;
                if (standstill_ctr_ > kStandstillTimeout)
                {
                    standstill_timed_out_ = true;
                }
            }
            else
            {
                standstill_ctr_ = 0;
            }
        }
    }

    void updateSensor(raylib::Image &render_buffer, std::vector<Vec2d> &hitpoints_out)
    {
        float ray_start_x, ray_start_y;
        if (is_cam_following_)
        {
            ray_start_x = kScreenWidth / 2.F + radius_ * cos(DEG2RAD * rot_);
            ray_start_y = kScreenHeight / 2.F + radius_ * -sin(DEG2RAD * rot_);
        }
        else
        {
            ray_start_x = pos_.x + radius_ * cos(DEG2RAD * rot_);
            ray_start_y = kScreenHeight - pos_.y + radius_ * -sin(DEG2RAD * rot_);
        }
        if (has_raycast_sensor_)
        {
            hitpoints_out.clear();
            for (const auto angle : sensor_ray_angles_)
            {
                const float end_pos_x = ray_start_x + sensor_range_ * 5.F * cos(DEG2RAD * (rot_ + angle));
                const float end_pos_y = ray_start_y + sensor_range_ * 5.F * -sin(DEG2RAD * (rot_ + angle));

                auto pixels_along_ray = okitch::bresenham(ray_start_x, ray_start_y, end_pos_x, end_pos_y);
                for (const auto pix : pixels_along_ray)
                {
                    // We check for the sensor hit based on the boundaries which are determined by the color
                    raylib::Color *pix_color =
                        &((raylib::Color *)render_buffer.data)[pix.y * render_buffer.width + pix.x];
                    if (isBoundaryPixel(*pix_color))
                    {
                        // Calculate hit point relative to the robot
                        Vec2d hit_pt_relative;
                        float xTranslated    = pix.x - ray_start_x;
                        float yTranslated    = pix.y - ray_start_y;
                        float angleInRadians = rot_ * DEG2RAD; // Convert angle to radians
                        hit_pt_relative.x    = xTranslated * cos(angleInRadians) - yTranslated * sin(angleInRadians);
                        hit_pt_relative.y    = xTranslated * sin(angleInRadians) + yTranslated * cos(angleInRadians);
                        hitpoints_out.push_back(hit_pt_relative);
                        break;
                    }
                    else if (kDrawSensorRays) // drawing ray paths
                    {
                        pix_color->r = 100;
                        pix_color->g = 100;
                        pix_color->b = 100;
                    }
                }
            }
        }
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot)
    {
        pos_          = reset_pos;
        rot_          = reset_rot;
        acceleration_ = 0.F;
        speed_        = 0.F;

        crashed_              = false;
        standstill_timed_out_ = false;
        standstill_ctr_       = 0;
    }

    raylib::Vector2 pos_{};
    float           speed_{};
    float           acceleration_{};
    float           rot_{};
    float           radius_{9.0F};
    int16_t         id_{};

    bool has_raycast_sensor_{false};
    bool manual_control_enabled_{false};
    bool auto_control_enabled_{false};
    bool is_cam_following_{false};

    std::vector<float>  sensor_ray_angles_;
    const raylib::Color ray_color_ = raylib::Color::Magenta();
    float               sensor_range_{80.0F};

    bool     crashed_{false};
    bool     standstill_timed_out_{false};
    uint32_t standstill_ctr_{0};

    static constexpr float kAccInc   = 5.0f;
    static constexpr float kRotSpeed = 5.0f;
};
} // namespace okitch