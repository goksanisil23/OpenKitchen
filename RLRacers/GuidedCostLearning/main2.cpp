#include "Agent.h"
#include "Environment/Environment.h"
#include <iostream>
#include <torch/torch.h>
#include <vector>

struct ActorCritic : torch::nn::Module
{
    static constexpr int64_t input_dim  = 7;
    static constexpr int64_t hidden_dim = 64;

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, actor_mean{nullptr}, critic{nullptr};
    torch::Tensor     log_std;

    ActorCritic()
        : fc1(register_module("fc1", torch::nn::Linear(input_dim, hidden_dim))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim))),
          actor_mean(register_module("actor_mean", torch::nn::Linear(hidden_dim, 2))),
          critic(register_module("critic", torch::nn::Linear(hidden_dim, 1))),
          // initialize log_std so initial std ≈ 20 (wide exploration)
          log_std(register_parameter("log_std", torch::full({2}, std::log(2.0f))))
    {
        // **initialize throttle bias to mid-range (≈50) so you move immediately**
        actor_mean->bias.data()[0] = 50.0f; // throttle
        actor_mean->bias.data()[1] = 0.0f;  // steering
    }

    // returns (action, log_prob, entropy, value)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> act(const torch::Tensor &state)
    {
        auto x = torch::relu(fc1(state));
        x      = torch::relu(fc2(x));

        auto mean = actor_mean(x);
        auto std  = torch::exp(log_std);

        // sample: mean + std * eps
        auto eps    = torch::randn_like(mean);
        auto action = mean + eps * std;

        // compute log_prob per-dim: -((a-m)^2)/(2σ^2) - log(σ) - 0.5·log(2π)
        auto var       = std * std;
        auto log_scale = log_std;
        auto log_prob  = -((action - mean).pow(2) / (2 * var)) - log_scale - 0.5 * std::log(2 * M_PI);
        log_prob       = log_prob.sum(-1); // scalar

        // entropy per-dim: 0.5 + 0.5·log(2π) + log(σ)
        auto entropy = (0.5 + 0.5 * std::log(2 * M_PI) + log_scale).sum(-1);

        auto value = critic(x).squeeze(-1);
        return {action, log_prob, entropy, value};
    }

    torch::Tensor value(const torch::Tensor &state)
    {
        auto x = torch::relu(fc1(state));
        x      = torch::relu(fc2(x));
        return critic(x).squeeze(-1);
    }
};

class GCLAgent : public Agent
{
  public:
    static constexpr float lr = 1e-4;

    GCLAgent(const Vec2d start_pos, const float start_rot, const int16_t id)
        : Agent(start_pos, start_rot, id), optimizer(actor_critic.parameters(), lr)
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

    void reset(const Vec2d &reset_pos, const float reset_rot) override
    {
        Agent::reset(reset_pos, reset_rot);
        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

    void updateAction() override
    {
        return;
    }

  public:
    ActorCritic        actor_critic;
    torch::optim::Adam optimizer;
};

// Get current state from agent's sensor hits
torch::Tensor getState(const GCLAgent &agent)
{
    std::vector<float> state_vals;
    constexpr float    kSensorRangeSquared = Agent::kSensorRange * Agent::kSensorRange;
    for (const auto &hit : agent.sensor_hits_)
    {
        state_vals.push_back(hit.norm() / Agent::kSensorRange); // Normalized [0,1]
    }
    return torch::tensor(state_vals);
}

void setAction(GCLAgent &agent, torch::Tensor action)
{
    float throttle                       = action[0].item<float>();
    float steering                       = action[1].item<float>();
    agent.current_action_.throttle_delta = std::clamp(throttle, 0.0f, 100.0f);
    agent.current_action_.steering_delta = std::clamp(steering, -10.0f, 10.0f);

    // constexpr float kMaxThrottleMag = 100.0f;
    // constexpr float kMaxSteeringMag = 10.0f;
    // float           throttle        = (action[0].item<float>() + 1.0f) / 2.0f * kMaxThrottleMag; // [-1,1] to [0,100]
    // float           steering        = action[1].item<float>() * kMaxSteeringMag;                 // [-1,1] to [-10,10]
    // agent.current_action_.throttle_delta = throttle;
    // agent.current_action_.steering_delta = steering;
}

int main()
{
    // Hyperparams
    const float gamma        = 0.99;
    const float entropy_beta = 0.01;
    const int   n_steps      = 200;
    const int   EPISODES     = 2000;

    std::vector<std::unique_ptr<GCLAgent>> agents;
    agents.push_back(std::make_unique<GCLAgent>(Vec2d{0, 0}, 0, 0));
    GCLAgent *agent = agents[0].get();

    Environment env("/home/s0001734/Downloads/racetrack-database/tracks/IMS.csv", createBaseAgentPtrs(agents));

    for (int ep = 0; ep < EPISODES; ++ep)
    {
        env.resetAgent(agent);
        env.step();
        torch::Tensor state    = getState(*agent);
        Vec2d         prev_pos = agent->pos_;

        std::vector<torch::Tensor> log_probs, values, entropies;
        std::vector<float>         rewards;
        bool                       done = false;

        int t = 0;
        for (; t < n_steps; ++t)
        {
            // 1) policy forward & sample
            auto [action, log_prob, entropy, value] = agent->actor_critic.act(state);
            log_probs.push_back(log_prob);
            values.push_back(value);
            entropies.push_back(entropy);

            // 2) Apply action
            setAction(*agent, action);
            env.step();
            if (agent->crashed_)
            {
                done = true;
            }
            const float dist = (agent->pos_ - prev_pos).norm();
            prev_pos         = agent->pos_;
            rewards.push_back(done ? -5.0f : dist);

            // 3) manual reward: +1 for every non‐crashing step
            if (done)
                break;
            else
                state = getState(*agent);
        }

        // Bootstrap value
        torch::Tensor R = done ? torch::zeros({1}) : agent->actor_critic.value(state).detach();
        // Compute returns & advantages
        std::vector<torch::Tensor> returns(t);
        for (int i = t - 1; i >= 0; --i)
        {
            R          = rewards[i] + gamma * R;
            returns[i] = R;
        }

        // Compute losses
        torch::Tensor actor_loss   = torch::zeros({1});
        torch::Tensor critic_loss  = torch::zeros({1});
        torch::Tensor entropy_loss = torch::zeros({1});

        for (int i = 0; i < t; ++i)
        {
            auto advantage = returns[i] - values[i];
            actor_loss += -log_probs[i] * advantage.detach();
            critic_loss += advantage.pow(2);
            entropy_loss += -entropies[i];
        }
        actor_loss /= static_cast<float>(t);
        critic_loss = critic_loss / static_cast<float>(t);
        entropy_loss /= static_cast<float>(t);
        auto loss = actor_loss + critic_loss + entropy_beta * entropy_loss;

        // Optimize
        agent->optimizer.zero_grad();
        loss.backward();
        agent->optimizer.step();

        if (ep % 10 == 0)
        {
            std::cout << "Ep " << ep << " | Loss: " << loss.item<float>()
                      << " | Reward: " << std::accumulate(rewards.begin(), rewards.end(), 0.0f) << "\n";
        }
    }
    std::cout << "Training finished.\n";

    // Use the trained policy to run the environment
    {
        torch::NoGradGuard no_grad;
        agent->actor_critic.eval();

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

            auto x    = torch::relu(agent->actor_critic.fc1(state));
            x         = torch::relu(agent->actor_critic.fc2(x));
            auto mean = agent->actor_critic.actor_mean(x);
            // here we just take the mean (deterministic policy)
            float raw_thr = mean[0].item<float>();
            float raw_str = mean[1].item<float>();
            // clamp into your valid range
            agent->current_action_.throttle_delta = std::clamp(raw_thr, 0.0f, 100.0f);
            agent->current_action_.steering_delta = std::clamp(raw_str, -10.0f, 10.0f);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // save your trained policy if you like
    return 0;
}
