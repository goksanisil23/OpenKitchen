#pragma once

#include <random>
#include <time.h>
#include <torch/torch.h>
#include <vector>

#include "CircularVector.hpp"
#include "Environment/Agent.h"

template <size_t Capacity, size_t StateDim, size_t ActionDim, typename ActionType, torch::Dtype TorchActionType>
struct ReplayBuffer
{
    typedef std::array<float, StateDim>       State;
    typedef std::array<ActionType, ActionDim> Action;

    CircularVector<State>  states      = CircularVector<State>(Capacity);
    CircularVector<State>  next_states = CircularVector<State>(Capacity);
    CircularVector<Action> actions     = CircularVector<Action>(Capacity);
    CircularVector<float>  rewards     = CircularVector<float>(Capacity);
    CircularVector<float>  dones       = CircularVector<float>(Capacity);

    struct Samples
    {
        torch::Tensor states;
        torch::Tensor next_states;
        torch::Tensor actions;
        torch::Tensor rewards;
        torch::Tensor dones;
    };

    Samples sample(uint16_t batch_size, torch::Device device = torch::kCPU)
    {
        auto const f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        Samples samples;

        static thread_local std::mt19937 rng{std::random_device{}()};

        const int                          replay_buffer_current_size{static_cast<int>(states.size())};
        std::uniform_int_distribution<int> dist(0, std::max(0, replay_buffer_current_size - 1));

        torch::Tensor chosen_states_tensor      = torch::empty({batch_size, StateDim}, f32);
        torch::Tensor chosen_next_states_tensor = torch::empty({batch_size, StateDim}, f32);
        torch::Tensor chosen_actions_tensor =
            torch::empty({batch_size, ActionDim}, torch::TensorOptions().dtype(TorchActionType).device(device));
        torch::Tensor chosen_rewards_tensor = torch::empty({batch_size, 1}, f32);
        torch::Tensor chosen_dones_tensor   = torch::empty({batch_size, 1}, f32);

        for (auto b = 0; b < batch_size; ++b)
        {
            const int idx = dist(rng);

            // states
            for (size_t i = 0; i < StateDim; ++i)
            {
                chosen_states_tensor[b][i]      = states[idx][i];
                chosen_next_states_tensor[b][i] = next_states[idx][i];
            }
            // actions
            for (size_t i = 0; i < ActionDim; ++i)
            {
                chosen_actions_tensor[b][i] = actions[idx][i];
            }
            chosen_rewards_tensor[b][0] = rewards[idx];
            chosen_dones_tensor[b][0]   = dones[idx];
        }

        samples.states      = chosen_states_tensor.clone();
        samples.next_states = chosen_next_states_tensor.clone();
        samples.actions     = chosen_actions_tensor.clone();
        samples.rewards     = chosen_rewards_tensor.clone();
        samples.dones       = chosen_dones_tensor.clone();

        return samples;
    }
};
