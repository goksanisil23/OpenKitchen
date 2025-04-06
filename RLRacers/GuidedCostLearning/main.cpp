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
    }

    // Create an input tensor to the network from the sensor measurement
    void stateToTensor()
    {
        std::vector<float> flat_sensor_hits;
        flat_sensor_hits.reserve(sensor_hits_.size() * 2);
        for (const auto &hit : sensor_hits_)
        {
            flat_sensor_hits.push_back(hit.x / Agent::kSensorRange);
            flat_sensor_hits.push_back(hit.y / Agent::kSensorRange);
        }
        current_state_tensor_ = torch::tensor(flat_sensor_hits);
    }

    void updateAction() override
    {
        auto action            = policy_net_.forward(current_state_tensor_.unsqueeze(0));
        current_action_tensor_ = action;

        current_action_.throttle_delta = action[0][0].detach().item<float>() * 50.F + 50.F; // [-1,1] -> [0,100]
        current_action_.steering_delta = action[0][1].detach().item<float>() * 10.F;        // [-1,1] -> [-10,10]
        // std::cout << "Throttle: " << current_action_.throttle_delta << " Steering: " << current_action_.steering_delta
        //           << std::endl;
    }

    void reset(const Vec2d &reset_pos, const float reset_rot, const size_t track_reset_idx)
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
};

int32_t pickResetPosition(const Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

int main(int argc, char **argv)
{
    auto const expert_dataset = load_dataset(
        "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions");

    auto const race_tracks = getRaceTrackFiles("/home/s0001734/Downloads/racetrack-database/tracks");

    const size_t               N              = 1000; // Samples per iteration from expert and policy
    const size_t               num_iterations = 100;
    std::vector<torch::Tensor> sampled_expert_states_vec;
    std::vector<torch::Tensor> sampled_expert_actions_vec;
    sampled_expert_states_vec.reserve(N);
    sampled_expert_actions_vec.reserve(N);
    std::vector<torch::Tensor> sampled_policy_states_vec;
    std::vector<torch::Tensor> sampled_policy_actions_vec;
    sampled_policy_states_vec.reserve(N);
    sampled_policy_actions_vec.reserve(N);

    std::vector<std::unique_ptr<GCLAgent>> agents;
    agents.push_back(std::make_unique<GCLAgent>(Vec2d{0, 0}, 0, 0));
    std::vector<Agent *> agent_ptrs;
    for (const auto &agent_ptr : agents)
    {
        agent_ptrs.push_back(agent_ptr.get());
    }

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        // 1) Randomly sample N expert samples
        sampled_expert_states_vec.clear();
        sampled_expert_actions_vec.clear();
        auto const expert_samples = random_sample(expert_dataset, N);
        for (auto const &sample : expert_samples)
        {
            sampled_expert_states_vec.push_back(sample.state);
            sampled_expert_actions_vec.push_back(sample.action);
        }
        auto expert_states  = torch::stack(sampled_expert_states_vec);
        auto expert_actions = torch::stack(sampled_expert_actions_vec);

        // 2) Collect N samples with the current policy from the environment (Rollout)
        const size_t race_track_id_to_use{iter % race_tracks.size()};
        Environment  env(race_tracks[race_track_id_to_use], agent_ptrs);
        const float  start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
        const float  start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
        agents[0]->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx], 0);
        int32_t reset_idx{RaceTrack::kStartingIdx};
        sampled_policy_states_vec.clear();
        sampled_policy_actions_vec.clear();
        for (size_t i = 0; i < N; i++)
        {
            if (agents[0]->crashed_)
            {
                reset_idx = pickResetPosition(env, agents[0].get());
                agents[0]->reset({env.race_track_->track_data_points_.x_m[reset_idx],
                                  env.race_track_->track_data_points_.y_m[reset_idx]},
                                 env.race_track_->headings_[reset_idx],
                                 env.race_track_->findNearestTrackIndexBruteForce(
                                     {env.race_track_->track_data_points_.x_m[reset_idx],
                                      env.race_track_->track_data_points_.y_m[reset_idx]}));
            }
            agents[0]->stateToTensor();
            sampled_policy_states_vec.push_back(agents[0]->current_state_tensor_);
            agents[0]->updateAction();
            sampled_policy_actions_vec.push_back(agents[0]->current_action_tensor_.squeeze(0));
            env.step(); // agent moves in the environment with current_action, produces next_state
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // 3) Reward network optimization
        agents[0]->reward_optimizer_.zero_grad();

        auto policy_states  = torch::stack(sampled_policy_states_vec);  // [N, 14]
        auto policy_actions = torch::stack(sampled_policy_actions_vec); // [N, 2]

        // Network outputs reward estimation per each sample, so we take a mean of N samples
        auto const expert_reward = agents[0]->reward_net_.forward(expert_states, expert_actions).mean();
        // auto const policy_reward = agents[0]->reward_net_.forward(policy_states, policy_actions).mean();
        auto const policy_reward =
            agents[0]->reward_net_.forward(policy_states.detach(), policy_actions.detach()).mean();

        // Purpose is to maximize differentiability power of the reward network by boosting expert rewards
        // and penalizing noob-policy rewards
        auto const reward_loss = -expert_reward + policy_reward;
        reward_loss.backward();
        agents[0]->reward_optimizer_.step();

        // 4) Policy network optimization
        agents[0]->policy_optimizer_.zero_grad();
        auto new_policy_actions = agents[0]->policy_net_.forward(policy_states);
        auto policy_reward_from_new_rewardnet =
            agents[0]->reward_net_.forward(policy_states, new_policy_actions).mean();
        // auto const policy_reward_from_new_rewardnet = agents[0]->reward_net_.forward(policy_states, policy_actions).mean();
        auto const policy_loss = -policy_reward_from_new_rewardnet;
        policy_loss.backward();
        agents[0]->policy_optimizer_.step();

        std::cout << "Iteration: " << iter << ", Reward Loss: " << reward_loss.item<float>()
                  << ", Policy Loss: " << policy_loss.item<float>() << std::endl;
    }
    return 0;
}