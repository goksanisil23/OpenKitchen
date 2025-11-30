#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "Environment/Environment.h"

constexpr int16_t kNumAgents{1};

class TemplateAgent : public Agent
{
  public:
    TemplateAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {

        device_ = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        std::cout << "Using device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Re-configure the sensor ray angles so that we only have 5 rays
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-70.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(70.F);

        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        auto cpu_view = torch::from_blob(sensor_hits_norm.data(),
                                         {static_cast<int64_t>(sensor_hits_norm.size())},
                                         torch::TensorOptions().dtype(torch::kFloat32));

        return cpu_view.clone().to(device_);
    }

    void updateAction() override
    {
        // TODO: Based on sensor readings
        current_state_tensor_ = stateToTensor();

        current_action_.throttle_delta = 10.F;
        current_action_.steering_delta = 0;
    }

  public:
    torch::Tensor current_state_tensor_;

    torch::Device device_{torch::kCPU};
};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    std::vector<std::unique_ptr<TemplateAgent>> agents;
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.push_back(std::make_unique<TemplateAgent>(Vec2d{0, 0}, 0, i));
    }

    Environment env(argv[1], createBaseAgentPtrs(agents));

    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    for (auto &agent : agents)
    {
        agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);
    }

    uint32_t episode_idx{0};

    bool all_done{false};
    while (true)
    {
        std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
        all_done = false;

        for (auto &agent : agents)
        {
            env.resetAgent(agent.get());
        }

        // need to get an initial observation for the intial action, after reset
        env.step();

        while (!all_done)
        {
            for (auto &agent : agents)
            {
                agent->updateAction();
            }

            env.step();

            all_done = true;
            for (auto &agent : agents)
            {
                if (!agent->crashed_)
                {
                    all_done = false;
                }
            }
        }

        episode_idx++;
    }

    return 0;
}