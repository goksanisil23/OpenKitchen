#include <iostream>

#include "Network.hpp"

constexpr int32_t STATE_SIZE{}; // size of the inputs to the network, the "state of the environment" in RL
constexpr int32_t HIDDEN_LAYER_SIZE{};
constexpr int32_t ACTION_SIZE{};

constexpr float ADAM_OPT_LEARNING_RATE{0.001};

constexpr int32_t NUM_EPISODES{100};

int main()
{
    Network            model(STATE_SIZE, HIDDEN_LAYER_SIZE, ACTION_SIZE);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(ADAM_OPT_LEARNING_RATE));

    // Loop through episodes
    for (int32_t episode = 0; episode < NUM_EPISODES; ++episode)
    {
        auto state = get_initial_state(); // Define this as per your environment

        int32_t action = select_action(state, model, epsilon);

        for (;;)
        {
            auto [next_state, reward, done] = step(action);

            int32_t next_action = select_action(next_state, model, epsilon);

            train(state, action, reward, next_state, next_action, model, optimizer, alpha, gamma);

            state  = next_state;
            action = next_action;

            if (done)
                break;
        }
    }

    return 0;
}
