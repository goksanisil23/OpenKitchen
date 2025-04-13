#pragma once

#undef NDEBUG

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>

#include "CNN.hpp"

#include "Environment/Agent.h"
#include "Environment/IpcMsgs.h"
#include "Environment/Typedefs.h"
#include "Environment/Utils.h"

namespace rl
{
class DQCNNAgent : public Agent
{
  public:
    static constexpr float kVelocity{60.0};
    static constexpr float kSteeringDelta{5}; // degrees

    static constexpr float kEpsilon{0.99};         // probability of choosing random action (initial value)
    static constexpr float kEpsilonDiscount{0.01}; // how much epsilon decreases to next episode
    static constexpr float kGamma{0.99};           // discount factor btw current and future rewards
    static constexpr float kLearningRate{1e-4};    // learning rate for the Adam optimizer

    static constexpr size_t kImageSize{CNN::kImageWidth * CNN::kImageHeight * CNN::kImageChannels};

    static constexpr size_t kStateDim{kImageSize};
    static constexpr size_t kActionDim{1}; // since we only choose index

    const std::unordered_map<int64_t, std::pair<float, float>> kActionMap{
        {0, {kVelocity, 0.F}},
        {1, {kVelocity / 2.F, kSteeringDelta}},
        {2, {kVelocity / 2.F, -kSteeringDelta}},
        {3, {kVelocity / 2.F, kSteeringDelta / 2.F}},
        {4, {kVelocity / 2.F, -kSteeringDelta / 2.F}}};

    DQCNNAgent() = default;

    // Used when all agents are created initially, with randomized weights
    DQCNNAgent(const Vec2d   start_pos,
               const float   start_rot,
               const int16_t id,
               const size_t  start_idx     = 0,
               const size_t  track_idx_len = 0)
        : Agent(start_pos, start_rot, id), prev_track_idx_{static_cast<int64_t>(start_idx)},
          track_idx_len_{static_cast<int64_t>(track_idx_len)}
    {
        assert(kActionMap.size() == CNN::kOutputSize);
    }

    // Convert the flat image array to a tensor
    torch::Tensor imageToTensor(const std::vector<uint8_t> &image_array)
    {
        // populate the input, but normalize the features so that all lie in [0,1]
        // Also downsample the image
        auto image_tensor = torch::from_blob((void *)(image_array.data()),
                                             {CNN::kImageHeight, CNN::kImageWidth, CNN::kImageChannels},
                                             torch::TensorOptions().dtype(torch::kUInt8));
        image_tensor      = image_tensor.to(torch::kFloat32).div(255.0);
        // Grayscale
        auto r                = image_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
        auto g                = image_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
        auto b                = image_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});
        auto grayscale_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        // Add the batch dimension and channel dimension for the grayscale image
        image_tensor = grayscale_tensor.unsqueeze(0).unsqueeze(0);
        // Downscale
        std::vector<int64_t> new_size = {CNN::kImageHeight / 4, CNN::kImageWidth / 4};
        image_tensor                  = torch::nn::functional::interpolate(
            image_tensor,
            torch::nn::functional::InterpolateFuncOptions().size(new_size).mode(torch::kBilinear).align_corners(false));

        // // Save
        // {
        //     std::cout << "image_tensor: " << image_tensor.sizes() << std::endl;
        //     auto save_tensor = image_tensor.squeeze();
        //     std::cout << "save_tensor: " << save_tensor.sizes() << std::endl;
        //     save_tensor = save_tensor.mul(255).clamp(0, 255).to(torch::kUInt8);

        //     // Save the tensor as a PPM image
        //     std::ofstream outfile("output.ppm", std::ios::out | std::ios::binary);
        //     outfile << "P6\n" << CNN::kImageWidth / 4 << " " << CNN::kImageHeight / 4 << "\n255\n";

        //     for (int y = 0; y < CNN::kImageHeight / 4; ++y)
        //     {
        //         for (int x = 0; x < CNN::kImageWidth / 4; ++x)
        //         {
        //             uint8_t pixel = save_tensor.index({y, x}).item<uint8_t>();
        //             outfile.write(reinterpret_cast<char *>(&pixel), 1);
        //             outfile.write(reinterpret_cast<char *>(&pixel), 1);
        //             outfile.write(reinterpret_cast<char *>(&pixel), 1);
        //         }
        //     }
        //     outfile.close();
        // }

        return image_tensor;
    }

    // Given the current state of the agent (measurements + internal states),
    // decide on the next action based on epsilon-greedy policy
    void updateAction() override
    {
        torch::NoGradGuard no_grad;

        const float rand_val = static_cast<float>(GetRandomValue(0, RAND_MAX)) / static_cast<float>(RAND_MAX);
        if (rand_val < epsilon_)
        {
            current_action_idx_ = static_cast<int64_t>(GetRandomValue(0, CNN::kOutputSize - 1));
        }
        else
        {
            std::cout << "FORWARD" << std::endl;
            std::cout << image_tensor_.sizes() << std::endl;
            auto q_vals_for_this_state = nn_.forward(image_tensor_);
            // Choose the action corresponding to maximum q-value estimate
            current_action_idx_ = q_vals_for_this_state.argmax().item<int64_t>();
        }

        auto const accel_steer_pair{kActionMap.at(current_action_idx_)};
        current_action_.throttle_delta = accel_steer_pair.first;
        current_action_.steering_delta = accel_steer_pair.second;

        // Add it to experience buffer
        saved_actions_.push_back(current_action_idx_);
    }

    static void updateDQN()
    {
        static constexpr int16_t kBatchSize{32};
        static constexpr size_t  kIterationSteps{4};

        std::cout << "Replay buffer size: " << saved_states_tensors_.size() << std::endl;
        for (size_t iter{0}; iter < kIterationSteps; iter++)
        {
            std::cout << "iteration: " << iter << std::endl;
            size_t num_batches = (saved_actions_.size() + kBatchSize - 1) / kBatchSize;
            for (size_t batch_idx{0}; batch_idx < num_batches; ++batch_idx)
            {
                std::cout << "batch: " << batch_idx << std::endl;

                size_t start_idx = batch_idx * kBatchSize;
                size_t end_idx   = std::min(start_idx + kBatchSize, saved_actions_.size());

                std::vector<torch::Tensor> sampled_states;
                std::vector<torch::Tensor> sampled_next_states;
                std::vector<torch::Tensor> sampled_rewards;
                std::vector<torch::Tensor> sampled_action_indices;

                for (size_t i = start_idx; i < end_idx; ++i)
                {
                    sampled_states.push_back(saved_states_tensors_[i]);
                    sampled_next_states.push_back(saved_next_states_tensors_[i]);
                    sampled_rewards.push_back(torch::tensor({saved_rewards_[i]}));
                    sampled_action_indices.push_back(
                        torch::tensor({static_cast<int64_t>(saved_actions_[i])}, torch::dtype(torch::kInt64)));
                }

                auto state        = torch::stack(sampled_states);
                auto next_state   = torch::stack(sampled_next_states);
                auto reward       = torch::stack(sampled_rewards);
                auto action_index = torch::stack(sampled_action_indices);

                auto q_values      = nn_.forward(state);
                auto next_q_values = nn_.forward(next_state).detach();

                // Both targets below works
                auto temporal_diff_target = reward + kGamma * torch::amax(next_q_values, 1, true);
                // auto temporal_diff_target = (reward + (1 - done) * kGamma * torch::amax(next_q_values, 1, true)).detach();

                // Create a target tensor
                auto target = q_values.clone().detach();

                for (int b{0}; b < target.size(0); ++b)
                {
                    target[b].index_put_({action_index[b].squeeze()}, temporal_diff_target[b].squeeze());
                }

                auto loss = torch::mse_loss(q_values, target);
                optimizer_.zero_grad(); // clear the previous gradients
                loss.backward();
                optimizer_.step();
            }
        }

        // Clear the experience buffer
        saved_states_tensors_.clear();
        saved_next_states_tensors_.clear();
        saved_actions_.clear();
        saved_rewards_.clear();
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot, const size_t track_reset_idx)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());

        prev_track_idx_ = track_reset_idx;
    }

    // Calculates the current reward of the agent
    float calculateReward(const size_t nearest_track_idx)
    {
        if (crashed_)
        {
            return -200.F;
        }

        float min_distance = Agent::kSensorRange;
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            if (min_distance > sensor_hits_[i].norm())
            {
                min_distance = sensor_hits_[i].norm();
            }
        }

        float progression_reward = min_distance;

        return progression_reward;
    }

  public:
    int64_t current_action_idx_;

    int64_t prev_track_idx_;
    int64_t track_idx_len_;

    float epsilon_{kEpsilon};

    torch::Tensor image_tensor_;

    static std::vector<torch::Tensor> saved_states_tensors_;
    static std::vector<torch::Tensor> saved_next_states_tensors_;
    static std::vector<size_t>        saved_actions_;
    static std::vector<float>         saved_rewards_;

    // Variables shared among all agents
    static CNN                nn_;
    static torch::optim::Adam optimizer_;
};

std::vector<torch::Tensor> DQCNNAgent::saved_states_tensors_{};
std::vector<torch::Tensor> DQCNNAgent::saved_next_states_tensors_{};
std::vector<size_t>        DQCNNAgent::saved_actions_{};
std::vector<float>         DQCNNAgent::saved_rewards_{};

CNN                DQCNNAgent::nn_{CNN()};
torch::optim::Adam DQCNNAgent::optimizer_{DQCNNAgent::nn_.parameters(), kLearningRate};

} // namespace rl