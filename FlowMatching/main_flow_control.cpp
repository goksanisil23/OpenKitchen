#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <torch/script.h>
#include <torch/torch.h>

#include "Environment/Environment.h"

constexpr bool kResetAgentsRandomly{true};
constexpr int  kNumEpisodes{300};

class FlowControlAgent : public Agent
{
  public:
    static constexpr int kDefaultImageSize = 128;
    static constexpr int kFlowSteps        = 32;

    FlowControlAgent(const Vec2d        start_pos,
                     float              start_rot,
                     int16_t            id,
                     const std::string &scripted_model_path,
                     int                image_size)
        : Agent(start_pos, start_rot, id), scripted_model_path_(scripted_model_path), image_size_(image_size)
    {
        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;
        setupModel();
    }

    void setupModel()
    {
        policy_model_ = torch::jit::load(scripted_model_path_);
        policy_model_.to(device_);
        policy_model_.eval();
    }

    torch::Tensor stateToTensor(const Environment &env)
    {
        using namespace torch::indexing;
        using namespace torch::nn::functional;

        const auto render_target_info = env.screen_grabber_->getRenderTargetInfo();

        // HxWx4 uint8 on CUDA
        auto t = torch::empty({render_target_info.height, render_target_info.width, render_target_info.channels},
                              torch::TensorOptions().dtype(torch::kUInt8).device(device_));

        const size_t dst_pitch = static_cast<size_t>(t.stride(0)) * t.itemsize();
        env.screen_grabber_->getRenderTargetDevice(t.data_ptr(), dst_pitch);

        // GPU transforms: flip, drop alpha, to float, NCHW, resize
        auto flipped = t.flip({0});
        auto rgb     = flipped.index({Slice(), Slice(), Slice(0, 3)});
        auto chw     = rgb.permute({2, 0, 1}).to(torch::kFloat).div_(255.0f);

        const torch::Tensor bev_input = interpolate(chw.unsqueeze(0),
                                                    InterpolateFuncOptions()
                                                        .size(std::vector<int64_t>{image_size_, image_size_})
                                                        .mode(torch::kBilinear)
                                                        .align_corners(false));

        return bev_input;
    }

    torch::Tensor sampleNormalizedAction(const torch::Tensor &image_tensor)
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat).device(device_);

        torch::Tensor x  = torch::randn({1, 2}, options);
        const float   dt = 1.0F / static_cast<float>(kFlowSteps);

        for (int i = 0; i < kFlowSteps; ++i)
        {
            const float         t_val = static_cast<float>(i) / static_cast<float>(kFlowSteps);
            const torch::Tensor t     = torch::full({1, 1}, t_val, options);

            const torch::Tensor v = policy_model_.forward({x, t, image_tensor}).toTensor();
            x                     = x + dt * v;
        }

        return x.clamp(-1.0, 1.0);
    }

    void updateAction(const Environment &env)
    {
        torch::NoGradGuard no_grad;

        const auto bev_input   = stateToTensor(env);
        const auto action_norm = sampleNormalizedAction(bev_input);

        const float throttle_norm = action_norm[0][0].item<float>();
        const float steering_norm = action_norm[0][1].item<float>();

        const float throttle = (throttle_norm + 1.0F) * 50.0F; // [-1,1] -> [0,100]
        const float steering = steering_norm * 10.0F;          // [-1,1] -> [-10,10]

        current_action_.throttle_delta = std::clamp(throttle, 0.0F, 100.0F);
        current_action_.steering_delta = std::clamp(steering, -10.0F, 10.0F);
    }

    void updateAction() override
    {
    }

    void reset(const Vec2d &reset_pos, float reset_rot) override
    {
        Agent::reset(reset_pos, reset_rot);
    }

  public:
    const torch::Device device_{torch::kCUDA};
    size_t              prev_track_idx_{};

  private:
    std::string                scripted_model_path_;
    int                        image_size_;
    torch::jit::script::Module policy_model_;
};

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0] << " <track_csv_path> <scripted_model_path> [image_size]\n";
        return -1;
    }

    if (!torch::cuda::is_available())
    {
        std::cerr << "CUDA is required.\n";
        return -1;
    }

    const std::string track_csv_path      = argv[1];
    const std::string scripted_model_path = argv[2];
    const int         image_size          = (argc == 4) ? std::stoi(argv[3]) : FlowControlAgent::kDefaultImageSize;

    std::vector<std::unique_ptr<FlowControlAgent>> agents;
    agents.push_back(std::make_unique<FlowControlAgent>(Vec2d{0, 0}, 0.0F, 0, scripted_model_path, image_size));

    constexpr bool kDrawRays{false};
    constexpr bool kHideWindow{false};
    Environment    env(track_csv_path, createBaseAgentPtrs(agents), kDrawRays, kHideWindow);

    env.visualizer_->setAgentToFollow(agents[0].get());

    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    auto       &agent = agents[0];
    agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);

    uint32_t episode_idx{0};

    while (episode_idx < kNumEpisodes)
    {
        env.resetAgent(agent.get(), kResetAgentsRandomly);

        env.step();
        agent->prev_track_idx_ = env.race_track_->findNearestTrackIndexBruteForce(agent->pos_);

        while (!agent->crashed_)
        {
            agent->updateAction(env);
            env.step();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        episode_idx++;
    }

    return 0;
}
