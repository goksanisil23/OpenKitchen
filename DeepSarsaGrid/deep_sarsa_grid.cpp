#include <array>
#include <raylib.h>
#include <torch/torch.h>

constexpr int screenWidth  = 800;
constexpr int screenHeight = 800;
constexpr int kGridSize    = 10;
constexpr int kCellSize    = screenWidth / kGridSize;

enum Action
{
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ACTION_COUNT
};

struct State
{
    int x, y;
};

struct Net : torch::nn::Module
{
    Net()
        : fc1(register_module("fc1", torch::nn::Linear(2, 24))), fc2(register_module("fc2", torch::nn::Linear(24, 24))),
          out(register_module("out", torch::nn::Linear(24, ACTION_COUNT)))
    {
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = out->forward(x);
        return x;
    }

    torch::nn::Linear fc1, fc2, out;
};

class SarsaAgent
{
  private:
    Net                q_network;
    torch::optim::Adam optimizer;
    double             epsilon;

  public:
    SarsaAgent() : optimizer(q_network.parameters(), /*lr=*/0.01), epsilon(0.9)
    {
    }

    Action chooseAction(const State &state)
    {
        if (torch::rand({1}).item<float>() < epsilon)
        {
            return static_cast<Action>(torch::randint(0, ACTION_COUNT, {1}).item<int>());
        }

        torch::Tensor state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        auto          q_values     = q_network.forward(state_tensor);
        return static_cast<Action>(q_values.argmax().item<int>());
    }

    void
    updateQNetwork(const State &state, Action action, double reward, const State &next_state, const Action &next_action)
    {
        torch::Tensor old_state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        torch::Tensor new_state_tensor = torch::tensor({next_state.x, next_state.y}, torch::kFloat32);

        auto old_q_values  = q_network.forward(old_state_tensor);
        auto next_q_values = q_network.forward(new_state_tensor);
        auto target        = old_q_values.clone().detach();
        target[action]     = reward + 0.9 * next_q_values[next_action].item<float>();

        optimizer.zero_grad();
        torch::mse_loss(old_q_values, target).backward();
        optimizer.step();
    }

    float getMaxQValue(const State &state)
    {
        torch::Tensor state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        auto          q_values     = q_network.forward(state_tensor);
        return q_values.max().item<float>();
    }
};

float getMaxValInQTable(const std::array<std::array<float, kGridSize>, kGridSize> &q_values_grid)
{
    float max_val{0.F};
    for (int i = 0; i < kGridSize; i++)
    {
        for (int j = 0; j < kGridSize; j++)
        {
            if (q_values_grid[i][j] > max_val)
            {
                max_val = q_values_grid[i][j];
            }
        }
    }
    return max_val;
}

void draw(const State                                               &agent_state,
          const Vector2                                             &goal,
          const std::array<std::array<float, kGridSize>, kGridSize> &q_values_grid)
{
    BeginDrawing();
    ClearBackground(RAYWHITE);

    float maxQVal = getMaxValInQTable(q_values_grid);
    for (int i = 0; i < kGridSize; i++)
    {
        for (int j = 0; j < kGridSize; j++)
        {
            Vector2 top_left       = {i * kCellSize, j * kCellSize};
            Vector2 bottom_right   = {(i + 1) * kCellSize, (j + 1) * kCellSize};
            uint8_t cell_intensity = static_cast<uint8_t>(255.F * q_values_grid[i][j] / maxQVal);
            DrawRectangle(top_left.x, top_left.y, kCellSize, kCellSize, (Color){0, 255, 0, cell_intensity});
            DrawRectangleLinesEx(
                {top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y}, 1, BLACK);
            char buffer[10];
            snprintf(buffer, sizeof(buffer), "%.2f", q_values_grid[i][j]);
            DrawText(buffer, i * kCellSize + 5, j * kCellSize + 5, 20, BLACK);
        }
    }

    DrawRectangle(goal.x * kCellSize, goal.y * kCellSize, kCellSize, kCellSize, (Color){0, 255, 0, 255}); //goal
    DrawText("GOAL", goal.x * kCellSize + kCellSize / 4, goal.y * kCellSize + kCellSize / 3, 20, RED);
    DrawRectangle(agent_state.x * kCellSize, agent_state.y * kCellSize, kCellSize, kCellSize, BLUE); // agent
    EndDrawing();
}

State move(const State &state, const Action &action)
{
    State next_state = state;
    switch (action)
    {
    case UP:
        if (state.y > 0)
        {
            next_state.y = state.y - 1;
        }
        else
        {
            next_state = state;
        }
        break;
    case DOWN:
        if (state.y < kGridSize - 1)
        {
            next_state.y = state.y + 1;
        }
        else
        {
            next_state = state;
        }
        break;
    case LEFT:
        if (state.x > 0)
        {
            next_state.x = state.x - 1;
        }
        else
        {
            next_state = state;
        }
        break;
    case RIGHT:
        if (state.x < kGridSize - 1)
        {
            next_state.x = state.x + 1;
        }
        else
        {
            next_state = state;
        }
        break;
    case ACTION_COUNT:
        std::runtime_error("");
        break;
    }

    return next_state;
}

int main(void)
{
    InitWindow(screenWidth, screenHeight, "Deep SARSA GridWorld with Torch");

    Vector2                                             goal  = {kGridSize - 1, kGridSize - 1};
    State                                               state = {0, 0};
    SarsaAgent                                          agent;
    std::array<std::array<float, kGridSize>, kGridSize> q_values_grid = {}; // initialize all Q-values to 0

    double reward;
    Action action, next_action;
    State  next_state;

    action = agent.chooseAction(state);

    bool        done    = false;
    const float maxDist = std::sqrt(std::pow(kGridSize - 1, 2) * 2);

    while (!WindowShouldClose())
    {
        draw(state, goal, q_values_grid);

        next_state = move(state, action);

        if (next_state.x == goal.x && next_state.y == goal.y)
        {
            done = true;
        }
        reward = maxDist - std::sqrt(std::pow(next_state.x - goal.x, 2) + std::pow(next_state.y - goal.y, 2));

        next_action = agent.chooseAction(next_state);

        agent.updateQNetwork(state, action, reward, next_state, next_action);

        q_values_grid[state.x][state.y] = agent.getMaxQValue(state);

        state  = next_state;
        action = next_action;

        if (done)
        {
            state = {0, 0};
            done  = false;
        }
    }

    CloseWindow();
    return 0;
}
