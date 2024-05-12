#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/torch.h>
#include <vector>

#include "CircularVector.hpp"
#include "Environment/Agent.h"

template <size_t StateDim, size_t ActionDim, typename ActionType, torch::Dtype TorchActionType>
struct ReplayBuffer
{
    typedef std::array<float, StateDim>       State;
    typedef std::array<ActionType, ActionDim> Action;

    static constexpr size_t kCapacity{1'000'000};

    CircularVector<State>  states      = CircularVector<State>(kCapacity);
    CircularVector<State>  next_states = CircularVector<State>(kCapacity);
    CircularVector<Action> actions     = CircularVector<Action>(kCapacity);
    CircularVector<float>  rewards     = CircularVector<float>(kCapacity);
    CircularVector<float>  dones       = CircularVector<float>(kCapacity);

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

        std::vector<int> chosen_idxs;
        chosen_idxs.reserve(batch_size);
        int replay_buffer_current_size{static_cast<int>(states.size())};

        for (auto i{0}; i < batch_size; i++)
        {
            int rand_idx = (rand() % replay_buffer_current_size);
            chosen_idxs.push_back(rand_idx);
        }

        torch::Tensor chosen_states_tensor      = torch::empty({batch_size, StateDim}, torch::kFloat32);
        torch::Tensor chosen_next_states_tensor = torch::empty({batch_size, StateDim}, torch::kFloat32);
        torch::Tensor chosen_actions_tensor =
            torch::empty({batch_size, ActionDim}, torch::TensorOptions().dtype(TorchActionType));
        torch::Tensor chosen_rewards_tensor = torch::empty({batch_size, 1}, torch::kFloat32);
        torch::Tensor chosen_dones_tensor   = torch::empty({batch_size, 1}, torch::kFloat32);

        // Flatten for torch conversion
        size_t batch_idx{0};
        for (auto idx : chosen_idxs)
        {
            chosen_states_tensor[batch_idx] = torch::from_blob(states[idx].data(), {StateDim}, torch::kFloat32);
            chosen_next_states_tensor[batch_idx] =
                torch::from_blob(next_states[idx].data(), {StateDim}, torch::kFloat32);
            chosen_actions_tensor[batch_idx] =
                torch::from_blob(&(actions[idx]), {ActionDim}, torch::TensorOptions().dtype(TorchActionType));
            chosen_rewards_tensor[batch_idx] = torch::from_blob(&(rewards[idx]), {1}, torch::kFloat32);
            chosen_dones_tensor[batch_idx]   = torch::from_blob(&(dones[idx]), {1}, torch::kFloat32);
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
