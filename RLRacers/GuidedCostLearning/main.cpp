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

#include "GCLAgent.hpp"
#include "ReadExpertData.hpp"

#include "Environment/Environment.h"

namespace
{
struct Trajectory
{
    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> actions;
};

std::vector<std::string> getRaceTrackFiles(const std::string &folder_path, const std::string track_name)
{
    namespace fs = std::filesystem;

    std::vector<std::string> track_files;
    for (const auto &entry : fs::directory_iterator(folder_path))
    {
        if (entry.path().extension() == ".csv")
        {
            // Skip the files that doesn't contain the track_name
            if ((track_name != "") && (entry.path().string().find(track_name) == std::string::npos))
            {
                continue;
            }
            track_files.push_back(entry.path().string());
        }
    }
    return track_files;
}

// Generate trajectory using the current policy
// Trajectory generateTrajectory(GCLAgent &agent, Environment &env, const size_t N)
// {
//     Trajectory trajectory;
//     trajectory.states.reserve(N);
//     trajectory.actions.reserve(N);

//     for (size_t n = 0; n < N; ++n)
//     {
//         env.step();
//         if (agent.crashed_)
//         {
//             env.resetAgent(&agent);
//             env.step();
//         }

//         torch::Tensor state = getState(agent);
//         trajectory.states.push_back(state);
//         torch::Tensor action = agent.policy_net_.forward(state).detach();
//         setAction(agent, action);
//         trajectory.actions.push_back(action);
//     }

//     return trajectory;
// }
Trajectory generateTrajectory(GCLAgent &agent, Environment &env, size_t N)
{
    Trajectory trajectory;
    trajectory.states.reserve(N);
    trajectory.actions.reserve(N);

    for (size_t i = 0; i < N; ++i)
    {
        env.step();
        if (agent.crashed_)
        {
            env.resetAgent(&agent);
            env.step();
        }

        // 1) get state
        auto s = getState(agent); // [7]

        // 2) policy → (mu, log_std)
        auto [mu, log_std] = agent.policy_net_.forward(s); // each [2]
        auto std           = torch::exp(log_std);

        // 3) sample in pre‑tanh space
        auto eps   = torch::randn_like(mu);
        auto a_raw = mu + std * eps;

        // 4) squash to [–1,1]
        auto a = torch::tanh(a_raw);

        // 5) apply
        setAction(agent, a);

        // 6) store for cost‐net update
        trajectory.states.push_back(s);
        trajectory.actions.push_back(a.detach());
    }

    return trajectory;
}

} // namespace

int main(int argc, char **argv)
{
    auto const race_tracks = getRaceTrackFiles("/home/s0001734/Downloads/racetrack-database/tracks", "SaoPaulo");
    std::cout << "num race tracks to use: " << race_tracks.size() << std::endl;

    auto const expert_dataset = loadDataset(
        "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions", "SAOPAULO");
    auto const all_expert_samples = getAllExpertSamplesInDataset(expert_dataset);
    auto const expert_states_all  = all_expert_samples.state;
    auto const expert_actions_all = all_expert_samples.action;

    const size_t N            = 2048; // Samples per episode from expert and policy
    const size_t num_episodes = 100;

    std::vector<std::unique_ptr<GCLAgent>> agents;
    agents.push_back(std::make_unique<GCLAgent>(Vec2d{0, 0}, 0, 0));
    GCLAgent *agent = agents[0].get();

    // // Behavioral Cloning: Initialize policy with expert data
    // {
    //     constexpr int num_bc_epochs = 400;
    //     for (int epoch = 0; epoch < num_bc_epochs; ++epoch)
    //     {
    //         torch::Tensor predicted_actions = agent->policy_net_.forward(expert_states_all);
    //         torch::Tensor loss              = torch::mse_loss(predicted_actions, expert_actions_all);
    //         agent->policy_optimizer_.zero_grad();
    //         loss.backward();
    //         agent->policy_optimizer_.step();
    //         if (epoch % 10 == 0)
    //         {
    //             std::cout << "BC Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    //         }
    //     }
    // }

    // GCL loop
    for (size_t episode = 0; episode < num_episodes; episode++)
    {
        // Create the environment
        const size_t race_track_id_to_use{episode % race_tracks.size()};
        Environment  env(race_tracks[race_track_id_to_use], createBaseAgentPtrs(agents));
        env.resetAgent(agent);

        // 0) Sample randomly from expert data
        auto const    expert_samples = sampleRandomExpertTrajectory(expert_dataset, N);
        torch::Tensor expert_states  = expert_samples.state;
        torch::Tensor expert_actions = expert_samples.action;

        // 1) Generate trajectory using the current policy
        Trajectory    policy_traj{generateTrajectory(*agent, env, N)};
        torch::Tensor policy_states  = torch::stack(policy_traj.states);
        torch::Tensor policy_actions = torch::stack(policy_traj.actions);

        // 2) Update cost network
        torch::Tensor cost_expert = agent->cost_net_.forward(expert_states, expert_actions);
        torch::Tensor cost_policy = agent->cost_net_.forward(policy_states, policy_actions);
        // Build labels
        auto expert_labels = torch::zeros_like(cost_expert);
        auto policy_labels = torch::ones_like(cost_policy);
        auto bce_loss      = torch::nn::BCEWithLogitsLoss();
        auto cost_loss     = bce_loss(cost_expert, expert_labels) + bce_loss(cost_policy, policy_labels);
        agent->cost_optimizer_.zero_grad();
        cost_loss.backward();
        agent->cost_optimizer_.step();

        // 3) Update policy network
        const float policy_loss_f = updatePolicy(*agent, env);

        // Logging
        {
            std::cout << "Episode " << episode << ", Cost Loss: " << cost_loss.item<float>()
                      << ", Policy Loss: " << policy_loss_f << std::endl;
        }
    }

    // Use the trained policy to run the environment
    {
        // Create the environment
        const size_t race_track_id_to_use{0};
        Environment  env(race_tracks[race_track_id_to_use], createBaseAgentPtrs(agents));
        env.resetAgent(agent);
        while (true)
        {
            env.step();
            if (agent->crashed_)
            {
                env.resetAgent(agent);
                env.step();
            }

            torch::Tensor state = getState(*agent);
            auto [mu, log_std]  = agent->policy_net_.forward(state);
            static_cast<void>(log_std);
            auto action = mu;
            setAction(*agent, action);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    return 0;
}