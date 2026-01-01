#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include "Environment/Environment.h"

constexpr bool kResetAgentsRandomly{true};

void tensor_to_bgr_u8(const torch::Tensor &y_nchw)
{
    // y: (1,3,H,W) float32 in [0,1]
    torch::Tensor t = y_nchw
                          .squeeze(0) // (3,H,W)
                          .clamp(0, 1)
                          .mul(255.0)
                          .to(torch::kUInt8)
                          .permute({1, 2, 0}) // (H,W,3) RGB
                          .contiguous()
                          .cpu();

    const int H = (int)t.size(0);
    const int W = (int)t.size(1);

    cv::Mat rgb(H, W, CV_8UC3, t.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    cv::imshow("y (resized)", bgr);
    cv::waitKey(0);
}

class DinoControlAgent : public Agent
{
  public:
    static constexpr int MODEL_INPUT_IMG_SIZE = 96;

    DinoControlAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;

        setupModel();
    }

    void setupModel()
    {
        bc_controller_model_ = torch::jit::load("../sao_paulo_policy_ts.pt");
        bc_controller_model_.to(device_);
        bc_controller_model_.eval();
    }

    torch::Tensor stateToTensor(const Environment &env)
    {
        using namespace torch::nn::functional;
        using namespace torch::indexing;

        const auto render_target_info = env.screen_grabber_->getRenderTargetInfo();
        // HxWx4 uint8 on CUDA
        auto t = torch::empty({render_target_info.height, render_target_info.width, render_target_info.channels},
                              torch::TensorOptions().dtype(torch::kUInt8).device(device_));
        const size_t dst_pitch = static_cast<size_t>(t.stride(0)) * t.itemsize(); // width*4
        env.screen_grabber_->getRenderTargetDevice(t.data_ptr(), dst_pitch);

        // GPU transforms: flip, drop alpha, to float, NCHW, resize
        auto flipped = t.flip({0});                                           // HxWx4
        auto rgb     = flipped.index({Slice(), Slice(), Slice(0, 3)});        // HxWx3 (RGBA->RGB)
        auto chw     = rgb.permute({2, 0, 1}).to(torch::kFloat).div_(255.0f); // 3xHxW

        const torch::Tensor model_input = interpolate(
            chw.unsqueeze(0),
            InterpolateFuncOptions().size(std::vector<int64_t>{MODEL_INPUT_IMG_SIZE, MODEL_INPUT_IMG_SIZE}));

        // tensor_to_bgr_u8(model_input);

        return model_input;
    }

    void updateAction(const Environment &env)
    {
        torch::NoGradGuard no_grad;
        auto const         model_input = stateToTensor(env);
        model_output_                  = bc_controller_model_.forward({model_input}).toTensor();

        updateAction();
    }

    void updateAction() override
    {

        current_action_.throttle_delta = 100.F;
        current_action_.steering_delta = model_output_[0][0].item<float>(); // [-1, 1]
        // Denormalize
        current_action_.steering_delta *= 10.F;
        // current_action_.steering_delta = std::clamp(current_action_.steering_delta, -10.F, 10.F);
    }

    void reset(const Vec2d &reset_pos, const float reset_rot) override
    {
        Agent::reset(reset_pos, reset_rot);
    }

  public:
    const torch::Device device_{torch::kCUDA};
    size_t              prev_track_idx_{};
    torch::Tensor       current_state_tensor_;

    torch::jit::script::Module bc_controller_model_;
    torch::Tensor              model_output_;
};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    std::vector<std::unique_ptr<DinoControlAgent>> agents;
    // We're gonna reuse the same agent for all the population (in serial)
    agents.push_back(std::make_unique<DinoControlAgent>(Vec2d{0, 0}, 0, 0));

    constexpr bool kDrawRays{false};
    constexpr bool kHideWindow{false};
    Environment    env(argv[1], createBaseAgentPtrs(agents), kDrawRays, kHideWindow);

    env.visualizer_->setAgentToFollow(agents[0].get());

    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    auto       &agent = agents[0];
    agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);

    uint32_t episode_idx{0};

    while (true)
    {
        env.resetAgent(agent.get(), kResetAgentsRandomly);

        // need to get an initial observation for the intial action, after reset
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
