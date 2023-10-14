// Utility functions specific to Evolutionary Learning Environment

#include "Environment/RaceTrack.hpp"
#include "GeneticAgent.hpp"

#include "raylib-cpp.hpp"

namespace genetic
{
namespace util
{
void drawLoadBar(const int32_t start_x,
                 const int32_t start_y,
                 const float   fill_ratio,
                 const int32_t bar_length,
                 const bool    active)
{
    DrawRectangle(start_x, start_y, bar_length, 12, BLACK);
    if (active)
    {
        const float bar_center{start_x + bar_length / 2.f};
        const float fill_amount{bar_length / 2.f * std::abs(fill_ratio)};
        if (fill_ratio > 0.F)
        {
            DrawRectangle(bar_center, start_y, fill_amount, 12, BLUE);
        }
        else
        {
            DrawRectangle(bar_center - fill_amount, start_y, fill_amount, 12, RED);
        }
    }

    DrawRectangleLines(start_x, start_y, bar_length, 12, DARKBROWN);
}

inline void drawEpisodeNum(const uint32_t episode_num)
{
    static Font custom_font = LoadFont("CooperHewitt-Semibold.otf");
    DrawTextEx(custom_font, TextFormat("Gen: %u", episode_num), {10, 40}, 17, 2, SKYBLUE);
}

// Draws actuation bars on the top right corner
inline void drawActionBar(const std::vector<GeneticAgent> &agents, const uint32_t iter)
{
    static Font       custom_font = LoadFont("CooperHewitt-Semibold.otf");
    constexpr int32_t kAccelBarLen(100);
    constexpr int32_t kSteerBarLen(100);
    constexpr int32_t kAccelBarStartX = kScreenWidth - kAccelBarLen;
    constexpr int32_t kSteerBarStartX = kScreenWidth - kAccelBarLen - kSteerBarLen - 5;
    constexpr int32_t kBarStartY      = 20;
    constexpr int32_t kBarSpacingY    = 10;
    constexpr int32_t kTextFontSize   = 13;
    DrawTextEx(custom_font, "Accel.", {kAccelBarStartX + kAccelBarLen / 2 - 10, 5}, kTextFontSize, 1, LIGHTGRAY);
    DrawTextEx(custom_font, "Steer.", {kSteerBarStartX + kSteerBarLen / 2 - 10, 5}, kTextFontSize, 1, LIGHTGRAY);
    for (size_t i{0}; i < agents.size(); i++)
    {
        DrawTextEx(custom_font,
                   TextFormat("%lu", i),
                   {kAccelBarStartX - 30, kBarStartY + static_cast<float>(i) * kBarSpacingY},
                   kTextFontSize,
                   1,
                   WHITE);
        drawLoadBar(kAccelBarStartX,
                    kBarStartY + i * kBarSpacingY,
                    agents[i].current_action_.acceleration_delta / genetic::GeneticAgent::kAccelerationDelta,
                    kAccelBarLen,
                    !agents[i].crashed_);
        constexpr float kSteerLimit{genetic::GeneticAgent::kSteeringDeltaHigh +
                                    genetic::GeneticAgent::kSteeringDeltaLow};
        drawLoadBar(kSteerBarStartX,
                    kBarStartY + i * kBarSpacingY,
                    agents[i].current_action_.steering_delta / kSteerLimit,
                    kSteerBarLen,
                    !agents[i].crashed_);
    }
    const float kIterTextPosY{kBarSpacingY * static_cast<float>(agents.size()) + kBarStartY + 7};
    DrawTextEx(custom_font,
               TextFormat("Step #: %u", iter),
               {(kAccelBarStartX + kSteerBarStartX) / 2.F + 10, kIterTextPosY},
               kTextFontSize,
               1,
               LIGHTGRAY);
}

// Draws the average colony score per episode
inline void showColonyScore(const std::vector<float> &colony_avg_scores)
{
    std::cout << "**** avg colony scores per episode ****" << std::endl;
    for (size_t i{0}; i < colony_avg_scores.size(); i++)
    {
        printf("%.2f ", colony_avg_scores[i]);
    }
    std::cout << std::endl;
}

void drawTrackPointNumbers(const RaceTrack::TrackData &track_data_points)
{
    for (size_t i{0}; i < track_data_points.x_m.size(); i++)
    {
        DrawText(TextFormat("%lu", i), track_data_points.x_m[i], track_data_points.y_m[i], 10, BLACK);
    }
}

void drawTrackTitle(const std::string &track_name)
{
    static Font custom_font = LoadFont("CooperHewitt-Semibold.otf");
    DrawTextEx(custom_font, track_name.c_str(), {kScreenWidth / 2, 10}, 25, 3, ORANGE);
}

float getAvgColonyScore(const std::vector<GeneticAgent> &agents)
{
    // Calculate the current average
    float current_avg{0.F};
    for (const auto &agent : agents)
    {
        current_avg += agent.score_;
    }
    current_avg /= static_cast<float>(agents.size());

    return current_avg;
}

void saveBestAgentNetwork(const std::vector<genetic::GeneticAgent> &agents)
{
    static float top_score_all_time{0.F};  // individual top score of an agent across all generations
    static float prev_gen_best_score{0.F}; // keeping track of the previous generation's best score

    size_t top_scorer_agent_id;
    float  best_score_current{0.F};
    for (size_t i{0}; i < agents.size(); i++)
    {

        if (agents[i].score_ > best_score_current)
        {
            best_score_current  = agents[i].score_;
            top_scorer_agent_id = i;
        }
    }

    if (prev_gen_best_score > best_score_current)
    {
        std::string error_string{"!! Current gen. high score " + std::to_string(best_score_current) +
                                 " is less than previous gen. high score " + std::to_string(prev_gen_best_score)};
        throw std::runtime_error(error_string);
        // std::cerr << error_string << std::endl;
    }
    prev_gen_best_score = best_score_current;

    if (best_score_current > top_score_all_time)
    {
        // Save the weights
        genetic::writeMatrixToFile("agent_weights_1.txt", agents[top_scorer_agent_id].nn_.weights_1_);
        genetic::writeMatrixToFile("agent_weights_2.txt", agents[top_scorer_agent_id].nn_.weights_2_);

        top_score_all_time = best_score_current;
    }

    // Compare previous action set
}

bool shouldResetEpisode(const std::vector<GeneticAgent> &agents)
{
    size_t num_drv_crashed{0};
    for (const auto &driver : agents)
    {
        if (driver.crashed_)
        {
            num_drv_crashed++;
        }
    }
    if (num_drv_crashed == agents.size())
    {
        return true;
    }
    return false;
}

void assignScores(std::vector<GeneticAgent> &agents, RaceTrack &race_track)
{
    for (auto &agent : agents)
    {
        auto nearest_track_idx = race_track.findNearestTrackIndexBruteForce({agent.pos_.x, agent.pos_.y});
        agent.score_           = static_cast<float>(nearest_track_idx);
    }
}
} // namespace util
} // namespace genetic