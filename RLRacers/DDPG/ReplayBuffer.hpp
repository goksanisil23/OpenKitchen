#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/torch.h>
#include <vector>

#include "CircularVector.hpp"
#include "Typedefs.h"

#include "Environment/Agent.h"

struct ReplayBuffer
{
    static constexpr size_t kCapacity{1'000'000};

    CircularVector<State>         states;
    CircularVector<State>         next_states;
    CircularVector<Agent::Action> actions;
    CircularVector<float>         rewards;
    CircularVector<float>         dones;

    ReplayBuffer()
    {
        states      = CircularVector<State>(kCapacity);
        next_states = CircularVector<State>(kCapacity);
        actions     = CircularVector<Agent::Action>(kCapacity);
        rewards     = CircularVector<float>(kCapacity);
        dones       = CircularVector<float>(kCapacity);
    }

    struct Samples
    {
        torch::Tensor states;
        torch::Tensor next_states;
        torch::Tensor actions;
        torch::Tensor rewards;
        torch::Tensor dones;
    };

    Samples sample(uint16_t batch_size)
    {
        Samples samples;
        srand(time(NULL));
        int upper = states.size() - 1;
        int lower = 0;

        std::vector<int> chosen_idxs;
        chosen_idxs.reserve(batch_size);

        for (auto i{0}; i < batch_size; i++)
        {
            int rand_idx = (rand() % (upper - lower + 1)) + lower;
            chosen_idxs.push_back(rand_idx);
        }

        torch::Tensor chosen_states_tensor      = torch::empty({batch_size, 5}); // 5 = state size
        torch::Tensor chosen_next_states_tensor = torch::empty({batch_size, 5});
        torch::Tensor chosen_actions_tensor     = torch::empty({batch_size, 2}); // 2 = actions size
        torch::Tensor chosen_rewards_tensor     = torch::empty({batch_size, 1});
        torch::Tensor chosen_dones_tensor       = torch::empty({batch_size, 1});

        // Flatten for torch conversion
        size_t batch_idx{0};
        for (auto idx : chosen_idxs)
        {
            chosen_states_tensor[batch_idx]      = torch::from_blob(states[idx].data(), {5}, torch::kFloat32);
            chosen_next_states_tensor[batch_idx] = torch::from_blob(next_states[idx].data(), {5}, torch::kFloat32);
            chosen_actions_tensor[batch_idx]     = torch::from_blob(&(actions[idx]), {2}, torch::kFloat32);
            chosen_rewards_tensor[batch_idx]     = torch::from_blob(&(rewards[idx]), {1}, torch::kFloat32);
            chosen_dones_tensor[batch_idx]       = torch::from_blob(&(dones[idx]), {1}, torch::kFloat32);
            batch_idx++;
        }

        samples.states      = chosen_states_tensor.clone();
        samples.next_states = chosen_next_states_tensor.clone();
        samples.actions     = chosen_actions_tensor.clone();
        samples.rewards     = chosen_rewards_tensor.clone();
        samples.dones       = chosen_dones_tensor.clone();

        return samples;
    }
};