#include <array>
#include <numeric>
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
bool operator==(const State &lhs, const State &rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

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

class DeepQAgent
{
  private:
    Net                q_network;
    torch::optim::Adam optimizer;
    double             epsilon;

  public:
    DeepQAgent() : optimizer(q_network.parameters(), /*lr=*/0.01), epsilon(0.9)
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

    void updateQNetwork(const State &state, Action action, double reward, const State &next_state, const bool done)
    {
        torch::Tensor old_state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        torch::Tensor new_state_tensor = torch::tensor({next_state.x, next_state.y}, torch::kFloat32);

        auto old_q_values  = q_network.forward(old_state_tensor);
        auto next_q_values = q_network.forward(new_state_tensor);
        auto target        = old_q_values.clone().detach();
        if (done)
            target[action] = reward;
        else
            target[action] = reward + 0.9 * next_q_values.max().item<float>();

        optimizer.zero_grad();
        torch::mse_loss(old_q_values, target).backward();
        optimizer.step();
    }

    std::pair<Action, float> getMaxQValue(const State &state)
    {
        torch::Tensor                            state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        auto                                     q_values     = q_network.forward(state_tensor);
        std::tuple<torch::Tensor, torch::Tensor> max_result   = q_values.max(0);
        float                                    max_value    = std::get<0>(max_result).item<float>();
        int32_t                                  max_index    = std::get<1>(max_result).item<int32_t>();
        return {static_cast<Action>(max_index), max_value};
    }
};

float getMaxValInQTable(const std::array<std::array<std::pair<Action, float>, kGridSize>, kGridSize> &q_values_grid)
{
    float max_val{std::numeric_limits<float>::lowest()};
    for (int i = 0; i < kGridSize; i++)
    {
        for (int j = 0; j < kGridSize; j++)
        {
            if (q_values_grid[i][j].second > max_val)
            {
                max_val = q_values_grid[i][j].second;
            }
        }
    }
    return max_val;
}

void drawArrow(Rectangle rect, Action dir)
{
    float   arrowLength = rect.width / 2;
    Vector2 middle      = {rect.x + rect.width / 2, rect.y + rect.height / 2};
    Vector2 startPoint, endPoint;

    switch (dir)
    {
    case UP:
        startPoint = {middle.x, middle.y};
        endPoint   = {middle.x, middle.y - arrowLength};
        break;
    case RIGHT:
        startPoint = {middle.x, middle.y};
        endPoint   = {middle.x + arrowLength, middle.y};
        break;
    case DOWN:
        startPoint = {middle.x, middle.y};
        endPoint   = {middle.x, middle.y + arrowLength};
        break;
    case LEFT:
        startPoint = {middle.x, middle.y};
        endPoint   = {middle.x - arrowLength, middle.y};
        break;
    case ACTION_COUNT:
        std::cout << "WRONG" << std::endl;
        break;
    }

    // Draw arrow line
    DrawLineV(startPoint, endPoint, BLACK);
}

void draw(const State                                                                  &agent_state,
          const State                                                                  &goal,
          const std::array<std::array<std::pair<Action, float>, kGridSize>, kGridSize> &q_values_grid)
{
    BeginDrawing();
    ClearBackground(RAYWHITE);

    const auto max_q_val = getMaxValInQTable(q_values_grid);
    for (int i = 0; i < kGridSize; i++)
    {
        for (int j = 0; j < kGridSize; j++)
        {
            Vector2 top_left       = {i * kCellSize, j * kCellSize};
            Vector2 bottom_right   = {(i + 1) * kCellSize, (j + 1) * kCellSize};
            uint8_t cell_intensity = static_cast<uint8_t>(255.F * ((q_values_grid[i][j].second) / max_q_val));
            DrawRectangle(top_left.x, top_left.y, kCellSize, kCellSize, (Color){0, 255, 0, cell_intensity});
            DrawRectangleLinesEx(
                {top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y}, 1, BLACK);
            char buffer[10];
            snprintf(buffer, sizeof(buffer), "%.2f", q_values_grid[i][j].second);
            DrawText(buffer, i * kCellSize + 5, j * kCellSize + 5, 15, BLACK);

            drawArrow({static_cast<float>(top_left.x),
                       static_cast<float>(top_left.y),
                       static_cast<float>(kCellSize),
                       static_cast<float>(kCellSize)},
                      q_values_grid[i][j].first);
        }
    }

    DrawRectangle(goal.x * kCellSize, goal.y * kCellSize, kCellSize, kCellSize, BLUE); //goal
    DrawText("GOAL", goal.x * kCellSize + kCellSize / 4, goal.y * kCellSize + kCellSize / 3, 20, RED);
    DrawRectangle(agent_state.x * kCellSize, agent_state.y * kCellSize, kCellSize, kCellSize, SKYBLUE); // agent
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
    // SetTargetFPS(10);

    State      goal  = {kGridSize - 1, kGridSize - 1};
    State      state = {0, 0};
    DeepQAgent agent;
    // best-action/value for a cell
    std::array<std::array<std::pair<Action, float>, kGridSize>, kGridSize> q_values_grid = {};

    double reward;
    Action action;
    State  next_state;

    action = agent.chooseAction(state);

    bool        done    = false;
    const float maxDist = std::sqrt(std::pow(kGridSize - 1, 2) * 2);

    while (!WindowShouldClose())
    {
        draw(state, goal, q_values_grid);

        action     = agent.chooseAction(state);
        next_state = move(state, action);

        // crashed the walls?
        if (next_state == state)
        {
            done   = true;
            reward = 0;
        }
        else if (next_state == goal)
        {
            done   = true;
            reward = maxDist - std::sqrt(std::pow(next_state.x - goal.x, 2) + std::pow(next_state.y - goal.y, 2));
        }
        else
        {
            reward = maxDist - std::sqrt(std::pow(next_state.x - goal.x, 2) + std::pow(next_state.y - goal.y, 2));
        }

        agent.updateQNetwork(state, action, reward, next_state, done);

        q_values_grid[state.x][state.y] = agent.getMaxQValue(state); // only for visualization

        state = next_state;

        if (done)
        {
            state  = {0, 0};
            done   = false;
            reward = 0.;
        }
    }

    CloseWindow();
    return 0;
}
