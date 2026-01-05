#pragma once

#include "Agent.h"
#include "Environment/Environment.h"
#include "Networks.hpp"

class GCLAgent : public Agent
{
  public:
    GCLAgent(const Vec2d start_pos, const float start_rot, const int16_t id)
        : Agent(start_pos, start_rot, id), cost_optimizer_(cost_net_.parameters(), torch::optim::AdamOptions(3e-4)),
          policy_optimizer_(policy_net_.parameters(), torch::optim::AdamOptions(3e-4)),
          value_optimizer_(value_net_.parameters(), torch::optim::AdamOptions(3e-4))
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
    CostNet   cost_net_;
    PolicyNet policy_net_;
    ValueNet  value_net_;

    torch::optim::Adam cost_optimizer_;
    torch::optim::Adam policy_optimizer_;
    torch::optim::Adam value_optimizer_;
};

// Get current state from agent's sensor hits
torch::Tensor getState(const GCLAgent &agent)
{
    std::vector<float> state_vals;
    constexpr float    kSensorRangeSquared = Agent::kSensorRange * Agent::kSensorRange;
    for (const auto &hit : agent.sensor_hits_)
    {
        state_vals.push_back(hit.squaredNorm() / kSensorRangeSquared); // Normalized [0,1]
    }
    return torch::tensor(state_vals);
}

// Set agent's current action from policy output
void setAction(GCLAgent &agent, torch::Tensor action)
{
    constexpr float kMaxThrottleMag = 100.0f;
    constexpr float kMaxSteeringMag = 10.0f;
    float           throttle        = (action[0].item<float>() + 1.0f) / 2.0f * kMaxThrottleMag; // [-1,1] to [0,100]
    float           steering        = action[1].item<float>() * kMaxSteeringMag;                 // [-1,1] to [-10,10]
    agent.current_action_.throttle_delta = throttle;
    agent.current_action_.steering_delta = steering;
}

// compute discounted returns Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + …
torch::Tensor computeDiscountedReturns(const std::vector<torch::Tensor> &rewards, float gamma = 0.99f)
{
    std::vector<torch::Tensor> returns(rewards.size());
    torch::Tensor              G = torch::zeros({1});
    for (int64_t i = static_cast<int64_t>(rewards.size() - 1); i >= 0; --i)
    {
        G          = rewards[i] + gamma * G;
        returns[i] = G;
    }
    return torch::stack(returns);
}

float updatePolicy(GCLAgent &agent, Environment &env)
{
    // hyperparameters
    constexpr size_t T        = 2048;
    constexpr float  gamma    = 0.99f;
    constexpr float  eps_clip = 0.2f;
    constexpr int    ACT_DIM  = 2; // action dimension

    // storage
    std::vector<torch::Tensor> states, actions_raw, logp_old, rewards;
    states.reserve(T);
    actions_raw.reserve(T);
    logp_old.reserve(T);
    rewards.reserve(T);

    // 1) Roll out T steps
    for (size_t t = 0; t < T; ++t)
    {
        env.step();
        if (agent.crashed_)
        {
            env.resetAgent(&agent);
            env.step();
        }

        auto s            = getState(agent);              // [7]
        auto [mu, logstd] = agent.policy_net_.forward(s); // [2], [2]
        auto std          = torch::exp(logstd);           // [2]

        // sample & squash
        auto eps    = torch::randn_like(mu); // [2]
        auto a_raw  = mu + std * eps;        // [2]
        auto a_tanh = torch::tanh(a_raw);    // [2]
        setAction(agent, a_tanh);

        // cost → reward
        auto c = agent.cost_net_.forward(s.unsqueeze(0), a_tanh.unsqueeze(0)).squeeze(0);

        // log‑prob under N(mu,std²):
        // –0.5 * ( ((a_raw–μ)/σ)² summed + 2·sum(logσ) + D·log(2π) )
        auto logp = -0.5 * (eps.pow(2).sum()   // sum over ACT_DIM
                            + 2 * logstd.sum() // sum over ACT_DIM
                            + ACT_DIM * std::log(2 * M_PI));

        states.push_back(s);
        actions_raw.push_back(a_raw);
        logp_old.push_back(logp.detach());
        rewards.push_back(-c.detach());
    }

    // 2) Discounted returns [T]
    auto returns = computeDiscountedReturns(rewards, gamma);

    // 3) Batch tensors
    auto states_b   = torch::stack(states);      // [T,7]
    auto actions_b  = torch::stack(actions_raw); // [T,2]
    auto logp_old_b = torch::stack(logp_old);    // [T]
    auto returns_b  = returns;                   // [T]

    // 4) Value predictions & advantages
    auto v_preds = agent.value_net_.forward(states_b); // [T]
    auto advs    = returns_b - v_preds.detach();       // [T]
    advs         = (advs - advs.mean()) / (advs.std() + 1e-8);

    // 5) Recompute log‑probs under current policy
    auto [mu_b, logstd_b] = agent.policy_net_.forward(states_b); // μ:[T,2], logstd:[2]
    auto std_b            = torch::exp(logstd_b);                // [2]
    auto eps_b            = (actions_b - mu_b) / std_b;          // [T,2]

    auto logp_b = -0.5 * (eps_b.pow(2).sum(1)            // [T], sum over ACT_DIM
                          + 2 * logstd_b.sum()           // scalar
                          + ACT_DIM * std::log(2 * M_PI) // scalar
                         );                              // [T]

    // 6) PPO clipped objective
    auto ratio   = torch::exp(logp_b - logp_old_b);                        // [T]
    auto surr1   = ratio * advs;                                           // [T]
    auto surr2   = torch::clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advs; // [T]
    auto loss_pi = -torch::mean(torch::min(surr1, surr2));                 // scalar

    // 7) Value loss
    auto loss_v = torch::mse_loss(v_preds, returns_b); // scalar

    // 8) Optimizer steps
    agent.policy_optimizer_.zero_grad();
    loss_pi.backward();
    agent.policy_optimizer_.step();

    agent.value_optimizer_.zero_grad();
    loss_v.backward();
    agent.value_optimizer_.step();

    return loss_pi.item<float>();
}
