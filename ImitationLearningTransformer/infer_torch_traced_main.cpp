#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

#include "Environment/Environment.h"

namespace
{
// Normalize a laser point (x, y) to [-1, 1]
std::vector<float> normalizeLaserPoint(float x, float y)
{
    constexpr float LASER_X_MIN{-Agent::kSensorRange};
    constexpr float LASER_X_MAX{Agent::kSensorRange};
    constexpr float LASER_Y_MIN{-Agent::kSensorRange};
    constexpr float LASER_Y_MAX{Agent::kSensorRange};

    float x_norm = 2 * (x - LASER_X_MIN) / (LASER_X_MAX - LASER_X_MIN) - 1;
    float y_norm = 2 * (y - LASER_Y_MIN) / (LASER_Y_MAX - LASER_Y_MIN) - 1;
    return {x_norm, y_norm};
}

// Denormalize controls (throttle in [0,100], steering in [-2,2])
std::vector<float> denormalizeControls(const std::vector<float> &norm)
{
    constexpr float THROTTLE_MIN{0.0f};
    constexpr float THROTTLE_MAX{100.0f};
    constexpr float STEERING_MAX{2.0f};
    constexpr float STEERING_MIN{-STEERING_MAX};

    float throttle = (norm[0] + 1.f) / 2.f * (THROTTLE_MAX - THROTTLE_MIN) + THROTTLE_MIN;
    float steering = (norm[1] + 1.f) / 2.f * (STEERING_MAX - STEERING_MIN) + STEERING_MIN;
    return {throttle, steering};
}

} // namespace

// Agent that loads the trained model as a TorchScript module and runs inference
class TracedInferAgent : public Agent
{
  public:
    TracedInferAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        // Setup sensor
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-90.F);
        sensor_ray_angles_.push_back(-60.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(60.F);
        sensor_ray_angles_.push_back(90.F);

        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;

        // Load the TorchScript model
        model_ = torch::jit::load(
            "/home/s0001734/Downloads/OpenKitchen/ImitationLearningTransformer/best_model_scripted.pt");
        model_.to(torch::kCUDA);
        model_.eval();

        normalized_rays_.reserve(sensor_ray_angles_.size() * 2);
    }
    void updateAction()
    {
        // Create an input tensor of shape [1, 7, 2]
        input_tensor_host_   = torch::from_blob(normalized_rays_.data(), {1, 7, 2}, torch::kFloat).clone();
        input_tensor_device_ = input_tensor_host_.to(torch::kCUDA);

        // Run inference
        output_tensor_device_ = model_.forward({input_tensor_device_}).toTensor();
        output_tensor_host_   = output_tensor_device_.to(torch::kCPU);

        // Extract and denormalize the predicted controls
        std::vector<float> normalized_control(output_tensor_host_.data_ptr<float>(),
                                              output_tensor_host_.data_ptr<float>() + output_tensor_host_.numel());
        std::vector<float> control = denormalizeControls(normalized_control);

        current_action_.throttle_delta = control[0];
        current_action_.steering_delta = control[1];
    }

    void normalizeRayMeasurements()
    {
        normalized_rays_.clear();
        for (const auto &hit : sensor_hits_)
        {
            std::vector<float> norm = normalizeLaserPoint(hit.x, hit.y);
            normalized_rays_.insert(normalized_rays_.end(), norm.begin(), norm.end());
        }
    }

  private:
    torch::jit::script::Module model_;
    std::vector<float>         normalized_rays_;

    torch::Tensor input_tensor_host_;
    torch::Tensor input_tensor_device_;
    torch::Tensor output_tensor_host_;
    torch::Tensor output_tensor_device_;
};

int32_t pickResetPosition(const Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a folder path for race track database\n";
        return -1;
    }

    std::vector<std::unique_ptr<TracedInferAgent>> agents;
    agents.push_back(std::make_unique<TracedInferAgent>(Vec2d{0, 0}, 0, 0));

    Environment       env(argv[1], createBaseAgentPtrs(agents));
    const std::string track_name{env.race_track_->track_name_};
    const float       start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float       start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (auto &agent : agents)
    {
        agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);
    }

    TracedInferAgent &agent = *agents[0];
    while (true)
    {
        env.resetAgent(&agent);
        while (!agent.crashed_)
        {
            env.step(); // agent moves in the environment with current_action, produces next_state
            agent.normalizeRayMeasurements();
            agent.updateAction();
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate some delay for rendering
        }
    }

    return 0;
}