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

    CircularVector<std::vector<float>> states;
    CircularVector<std::vector<float>> next_states;
    CircularVector<Agent::Action>      actions;
    CircularVector<float>              rewards;
    CircularVector<float>              dones;

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
        int upper = states.size();
        int lower = 0;

        std::vector<int> chosen_idxs;
        chosen_idxs.reserve(batch_size);

        for (auto i{0}; i < batch_size; i++)
        {
            int rand_idx = (rand() % (upper - lower + 1)) + lower;
            chosen_idxs.push_back(rand_idx);
        }

        std::vector<float> chosen_states_flat;
        std::vector<float> chosen_next_states_flat;
        std::vector<float> chosen_actions_flat;
        std::vector<float> chosen_rewards_flat;
        std::vector<float> chosen_dones_flat;

        chosen_states_flat.reserve(batch_size * states[0].size());
        chosen_next_states_flat.reserve(batch_size * next_states[0].size());
        chosen_actions_flat.reserve(batch_size * 2);
        chosen_rewards_flat.reserve(batch_size);
        chosen_dones_flat.reserve(batch_size);

        // Flatten for torch conversion
        for (auto idx : chosen_idxs)
        {
            const auto &chosen_states = states[idx];
            chosen_states_flat.insert(chosen_states_flat.end(), chosen_states.begin(), chosen_states.end());

            const auto &chosen_next_states = next_states[idx];
            chosen_next_states_flat.insert(
                chosen_next_states_flat.end(), chosen_next_states.begin(), chosen_next_states.end());

            chosen_actions_flat.push_back(actions[idx].acceleration_delta);
            chosen_actions_flat.push_back(actions[idx].steering_delta);

            chosen_rewards_flat.push_back(rewards[idx]);

            chosen_dones_flat.push_back(dones[idx]);
        }

        torch::Tensor chosen_states_tensor =
            torch::from_blob(chosen_states_flat.data(), {static_cast<int64_t>(states[0].size()), batch_size});
        torch::Tensor chosen_next_states_tensor =
            torch::from_blob(chosen_next_states_flat.data(), {static_cast<int64_t>(next_states[0].size()), batch_size});
        torch::Tensor chosen_actions_tensor =
            torch::from_blob(chosen_actions_flat.data(), {static_cast<int64_t>(2), batch_size});
        torch::Tensor chosen_rewards_tensor =
            torch::from_blob(chosen_rewards_flat.data(), {static_cast<int64_t>(1), batch_size});
        torch::Tensor chosen_dones_tensor =
            torch::from_blob(chosen_dones_flat.data(), {static_cast<int64_t>(1), batch_size});

        samples.states      = chosen_states_tensor.clone();
        samples.next_states = chosen_next_states_tensor.clone();
        samples.actions     = chosen_actions_tensor.clone();
        samples.rewards     = chosen_rewards_tensor.clone();
        samples.dones       = chosen_dones_tensor.clone();

        return samples;
    }
};