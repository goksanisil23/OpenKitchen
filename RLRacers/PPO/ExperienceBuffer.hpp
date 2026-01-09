#pragma once

#include <torch/torch.h>
#include <vector>

struct ExperienceBuffer
{
    std::vector<torch::Tensor> saved_log_probs;
    std::vector<torch::Tensor> saved_states;
    std::vector<float>         saved_rewards;
    std::vector<size_t>        saved_actions;

    torch::Tensor disc_rewards_tensor;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(const size_t batch_size,
                                                                                  const size_t index_offset)
    {
        assert(disc_rewards_tensor.numel() != 0);

        std::vector<size_t>        actions_batch(batch_size);
        std::vector<torch::Tensor> log_probs_batch(batch_size);
        std::vector<torch::Tensor> states_batch(batch_size);
        size_t                     buffer_idx{index_offset};
        for (size_t i = 0; i < batch_size; ++i)
        {
            actions_batch[i]   = saved_actions[buffer_idx];
            log_probs_batch[i] = saved_log_probs[buffer_idx];
            states_batch[i]    = saved_states[buffer_idx];
            buffer_idx++;
        }
        torch::Tensor actions_tensor = torch::from_blob(actions_batch.data(),
                                                        {static_cast<int64_t>(batch_size)},
                                                        torch::TensorOptions().dtype(torch::kInt64))
                                           .clone()
                                           .unsqueeze(1);
        torch::Tensor log_probs_tensor = torch::stack(log_probs_batch, 0).unsqueeze(1);
        torch::Tensor states_tensor    = torch::stack(states_batch, 0);

        return std::make_tuple(disc_rewards_tensor.slice(0, index_offset, index_offset + batch_size).unsqueeze(1),
                               actions_tensor,
                               log_probs_tensor,
                               states_tensor);
    }

    void calculateDiscountedRewards()
    {
        static constexpr float kGamma{0.99}; // discount factor btw current and future rewards
        constexpr float        kEps = std::numeric_limits<float>::epsilon();

        assert(disc_rewards_tensor.numel() == 0);

        std::vector<float> discounted_rewards(saved_rewards.size());
        float              cumulative_discounted_reward{0.F};

        // Reverse-iterate through rewards and calculate discounted returns
        size_t i = saved_rewards.size() - 1;
        for (auto r = saved_rewards.rbegin(); r != saved_rewards.rend(); ++r)
        {
            cumulative_discounted_reward = *r + kGamma * cumulative_discounted_reward;
            discounted_rewards[i]        = cumulative_discounted_reward;
            i--;
        }

        // Normalize the discounted rewards
        disc_rewards_tensor = torch::tensor(discounted_rewards);
        disc_rewards_tensor =
            (disc_rewards_tensor - disc_rewards_tensor.mean()) / (disc_rewards_tensor.std() + kEps).clone();
    }

    void clear()
    {
        saved_log_probs.clear();
        saved_states.clear();
        saved_rewards.clear();
        saved_actions.clear();

        disc_rewards_tensor = torch::Tensor();
    }
};