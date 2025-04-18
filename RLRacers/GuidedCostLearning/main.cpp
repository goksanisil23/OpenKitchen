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

#include "Networks.hpp"
#include "ReadExpertData.hpp"

#include "Environment/Environment.h"

namespace
{
class GCLAgent : public Agent
{
  public:
    GCLAgent(const Vec2d start_pos, const float start_rot, const int16_t id)
        : Agent(start_pos, start_rot, id),
          reward_optimizer_(torch::optim::Adam(reward_net_.parameters(), torch::optim::AdamOptions(1e-3))),
          policy_optimizer_(torch::optim::Adam(policy_net_.parameters(), torch::optim::AdamOptions(1e-4)))
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

        flat_sensor_hits_.reserve(sensor_hits_.size());
    }

    // Create an input tensor to the network from the sensor measurement
    void stateToTensor()
    {
        flat_sensor_hits_.clear();
        for (const auto &hit : sensor_hits_)
        {
            flat_sensor_hits_.push_back(hit.norm() / Agent::kSensorRange);
        }
        current_state_tensor_ = torch::tensor(flat_sensor_hits_);
    }

    void updateAction() override
    {
        auto action            = policy_net_.forward(current_state_tensor_.unsqueeze(0));
        current_action_tensor_ = action;

        current_action_.throttle_delta = action[0][0].item<float>() * 50.F + 50.F; // [-1,1] -> [0,100]
        current_action_.steering_delta = action[0][1].item<float>() * 10.F;        // [-1,1] -> [-10,10]
    }

    void reset(const Vec2d &reset_pos, const float reset_rot)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

  public:
    torch::Tensor current_state_tensor_;
    torch::Tensor current_action_tensor_;

    RewardNet reward_net_;
    PolicyNet policy_net_;

    torch::optim::Adam reward_optimizer_;
    torch::optim::Adam policy_optimizer_;

    std::vector<float> flat_sensor_hits_;
};

std::vector<std::string> getRaceTrackFiles(const std::string &folder_path)
{
    namespace fs = std::filesystem;

    std::vector<std::string> track_files;
    for (const auto &entry : fs::directory_iterator(folder_path))
    {
        if (entry.path().extension() == ".csv")
        {
            track_files.push_back(entry.path().string());
        }
    }
    return track_files;
}
} // namespace

int32_t pickResetPosition(const Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

int main(int argc, char **argv)
{
    auto const expert_dataset = load_dataset(
        "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions");

    auto const race_tracks = getRaceTrackFiles("/home/s0001734/Downloads/racetrack-database/tracks");

    const size_t               N            = 1'000; // Samples per episode from expert and policy
    const size_t               num_episodes = 500;
    std::vector<torch::Tensor> sampled_expert_states_vec;
    std::vector<torch::Tensor> sampled_expert_actions_vec;
    std::vector<torch::Tensor> sampled_policy_states_vec;
    std::vector<torch::Tensor> sampled_policy_actions_vec;
    sampled_expert_states_vec.reserve(N);
    sampled_expert_actions_vec.reserve(N);
    sampled_policy_states_vec.reserve(N);
    sampled_policy_actions_vec.reserve(N);

    std::vector<std::unique_ptr<GCLAgent>> agents;
    agents.push_back(std::make_unique<GCLAgent>(Vec2d{0, 0}, 0, 0));
    GCLAgent *agent = agents[0].get();

    for (size_t episode = 0; episode < num_episodes; episode++)
    {
        sampled_expert_states_vec.clear();
        sampled_expert_actions_vec.clear();
        sampled_policy_states_vec.clear();
        sampled_policy_actions_vec.clear();

        // 1) Randomly sample N expert samples
        auto const expert_samples = random_sample(expert_dataset, N);
        for (auto const &sample : expert_samples)
        {
            sampled_expert_states_vec.push_back(sample.state);
            sampled_expert_actions_vec.push_back(sample.action);
        }
        auto expert_states  = torch::stack(sampled_expert_states_vec);
        auto expert_actions = torch::stack(sampled_expert_actions_vec);

        // 2) Collect N samples with the current policy from the environment (Rollout)
        const size_t race_track_id_to_use{episode % race_tracks.size()};
        Environment  env(race_tracks[race_track_id_to_use], createBaseAgentPtrs(agents));
        const float  start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
        const float  start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
        agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);
        int32_t reset_idx{RaceTrack::kStartingIdx};

        for (size_t i = 0; i < N; i++)
        {
            if (agent->crashed_)
            {
                reset_idx = pickResetPosition(env, agent);
                agent->reset({env.race_track_->track_data_points_.x_m[reset_idx],
                              env.race_track_->track_data_points_.y_m[reset_idx]},
                             env.race_track_->headings_[reset_idx]);
            }

            env.step();

            agent->stateToTensor();
            sampled_policy_states_vec.push_back(agent->current_state_tensor_);
            agent->updateAction();
            sampled_policy_actions_vec.push_back(agent->current_action_tensor_.squeeze(0));
        }

        // 3) Reward network optimization
        agent->reward_optimizer_.zero_grad();

        auto policy_states  = torch::stack(sampled_policy_states_vec);  // [N, 7]
        auto policy_actions = torch::stack(sampled_policy_actions_vec); // [N, 2]

        // Network outputs reward estimation per each sample, so we take a mean of N samples
        auto const expert_reward = agent->reward_net_.forward(expert_states, expert_actions).mean();
        auto const policy_reward = agent->reward_net_.forward(policy_states, policy_actions).mean();

        // Purpose is to maximize differentiability power of the reward network by boosting expert rewards
        // and penalizing noob-policy rewards
        auto const reward_loss = -expert_reward + policy_reward;
        reward_loss.backward();
        agent->reward_optimizer_.step();

        // 4) Policy network optimization
        for (auto &param : agent->reward_net_.parameters())
            param.requires_grad_(false);
        agent->policy_optimizer_.zero_grad();
        auto new_policy_actions = agent->policy_net_.forward(policy_states);

        auto policy_reward_from_new_rewardnet = agent->reward_net_.forward(policy_states, new_policy_actions).mean();
        auto const policy_loss                = -policy_reward_from_new_rewardnet;
        policy_loss.backward();
        agent->policy_optimizer_.step();
        for (auto &param : agent->reward_net_.parameters())
            param.requires_grad_(true);

        std::cout << "episode: " << episode << ", Reward Loss: " << reward_loss.item<float>()
                  << ", Policy Loss: " << policy_loss.item<float>() << std::endl;
    }
    std::cout << "---- DONE ----" << std::endl;

    // Load a random track and run the agent
    {
        // Turn off gradients of the networks
        agent->reward_net_.eval();
        agent->policy_net_.eval();
        agent->reward_optimizer_.zero_grad();
        agent->policy_optimizer_.zero_grad();

        const size_t race_track_id_to_use{3};
        Environment  env(race_tracks[race_track_id_to_use], createBaseAgentPtrs(agents));
        const float  start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
        const float  start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
        agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);
        int32_t reset_idx{RaceTrack::kStartingIdx};

        while (true)
        {
            if (agent->crashed_)
            {
                reset_idx = pickResetPosition(env, agent);
                agent->reset({env.race_track_->track_data_points_.x_m[reset_idx],
                              env.race_track_->track_data_points_.y_m[reset_idx]},
                             env.race_track_->headings_[reset_idx]);
            }

            env.step(); // agent moves in the environment with current_action, produces next_state
            agent->stateToTensor();
            agent->updateAction();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    return 0;
}