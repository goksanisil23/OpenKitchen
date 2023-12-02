#include <array>
#include <numeric>
#include <raylib.h>
#include <torch/torch.h>

constexpr int            screenWidth  = 800;
constexpr int            screenHeight = 800;
constexpr int            kGridSize    = 10;
static constexpr int64_t kHiddenLayerSize{20};
constexpr int            kCellSize = screenWidth / kGridSize;
constexpr bool           kRandomReInit{true};

enum Action
{
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ACTION_COUNT
};

enum AgentType
{
    DeepSarsa,
    DeepQ
};

struct State
{
    int32_t x, y;
};

bool operator==(const State &lhs, const State &rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

struct Net : torch::nn::Module
{
    Net()
        : fc1(register_module("fc1", torch::nn::Linear(2, kHiddenLayerSize))),
          fc2(register_module("fc2", torch::nn::Linear(kHiddenLayerSize, kHiddenLayerSize))),
          out(register_module("out", torch::nn::Linear(kHiddenLayerSize, ACTION_COUNT)))
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

class DeepRlAgent
{
  public:
    static constexpr AgentType kAgentType{AgentType::DeepQ};
    static constexpr double    kLearningRate{0.001};
    static constexpr double    kEpsilon{0.99};
    static constexpr double    kEpsilonDecay{0.001};

    double               epsilon;
    std::shared_ptr<Net> q_network;
    torch::optim::Adam   optimizer;

  public:
    DeepRlAgent()
        : epsilon(kEpsilon), q_network(std::make_shared<Net>()), optimizer(q_network->parameters(), kLearningRate)
    {
    }

    Action chooseAction(const State &state)
    {
        if (torch::rand({1}).item<float>() < epsilon)
        {
            return static_cast<Action>(torch::randint(0, ACTION_COUNT, {1}).item<int>());
        }

        torch::Tensor state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        auto          q_values     = q_network->forward(state_tensor);
        return static_cast<Action>(q_values.argmax().item<int>());
    }

    void updateQNetwork(const State  &state,
                        Action        action,
                        double        reward,
                        const State  &next_state,
                        const Action &next_action,
                        const bool    done)
    {
        torch::Tensor old_state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        torch::Tensor new_state_tensor = torch::tensor({next_state.x, next_state.y}, torch::kFloat32);

        auto old_q_values = q_network->forward(old_state_tensor);
        auto target       = old_q_values.clone().detach();

        if constexpr (kAgentType == AgentType::DeepSarsa)
        {
            auto next_q_values = q_network->forward(new_state_tensor);
            if (done)
                target[action] = reward;
            else
                target[action] = reward + 0.9 * next_q_values[next_action].item<float>();
        }
        if constexpr (kAgentType == AgentType::DeepQ)
        // DQN
        {
            static_cast<void>(next_action);
            auto next_q_values     = q_network->forward(new_state_tensor);
            auto max_next_q_values = std::get<0>(next_q_values.max(0)); // get the max q-value
            if (done)
                target[action] = reward;
            else
                target[action] = reward + 0.9 * max_next_q_values;
        }
        optimizer.zero_grad();
        torch::mse_loss(old_q_values, target).backward();
        optimizer.step();
    }

    std::pair<Action, float> getMaxQValue(const State &state)
    {
        torch::Tensor                            state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        auto                                     q_values     = q_network->forward(state_tensor);
        std::tuple<torch::Tensor, torch::Tensor> max_result   = q_values.max(0);
        float                                    max_value    = std::get<0>(max_result).item<float>();
        int32_t                                  max_index    = std::get<1>(max_result).item<int32_t>();
        return {static_cast<Action>(max_index), max_value};
    }

    std::array<float, 4> getQValues(const State &state) const
    {
        torch::Tensor        state_tensor = torch::tensor({state.x, state.y}, torch::kFloat32);
        auto                 q_values     = q_network->forward(state_tensor);
        std::array<float, 4> res;
        for (int32_t i{0}; i < 4; i++)
        {
            res[i] = q_values[i].item<float>();
        }
        return res;
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

void resetState(State &state)
{
    if constexpr (kRandomReInit)
    {
        state = State{torch::randint(0, kGridSize - 1, {1}).item<int32_t>(),
                      torch::randint(0, kGridSize - 1, {1}).item<int32_t>()};
    }
    else
    {
        state = State{0, 0};
    }
}

void showAllQValues(const DeepRlAgent &agent)
{
    for (int i = 0; i < kGridSize; i++)
    {
        for (int j = 0; j < kGridSize; j++)
        {
            auto res = agent.getQValues({i, j});
            std::cout << "(" << i << "," << j << "): "
                      << "UP " << res[0] << " DOWN " << res[1] << " LEFT " << res[2] << " RIGHT " << res[3]
                      << std::endl;
        }
    }
}

int main(void)
{
    InitWindow(screenWidth, screenHeight, "Deep SARSA GridWorld with Torch");
    // SetTargetFPS(10);

    State goal = {kGridSize - 1, kGridSize - 1};
    State state;
    resetState(state);

    DeepRlAgent agent;
    // best-action/value for a cell
    std::array<std::array<std::pair<Action, float>, kGridSize>, kGridSize> q_values_grid = {};

    double reward;
    Action action, next_action;
    State  next_state;

    action = agent.chooseAction(state);

    bool        done    = false;
    const float maxDist = std::sqrt(std::pow(kGridSize - 1, 2) * 2);

    size_t ctr{0}, crash_ctr{0}, goal_reach_ctr{0};
    while (!WindowShouldClose())
    {
        ctr++;
        draw(state, goal, q_values_grid);

        next_state = move(state, action);

        // crashed the walls?
        if (next_state == state)
        {
            done   = true;
            reward = 0;
            crash_ctr++;
        }
        else if (next_state == goal)
        {
            done = true;
            // reward = maxDist - std::sqrt(std::pow(next_state.x - goal.x, 2) + std::pow(next_state.y - goal.y, 2));
            reward = maxDist;
            goal_reach_ctr++;
        }
        else
        {
            // reward = 0;
            reward = 1;
            // reward = maxDist - std::sqrt(std::pow(next_state.x - goal.x, 2) + std::pow(next_state.y - goal.y, 2));
        }

        next_action = agent.chooseAction(next_state);

        agent.updateQNetwork(state, action, reward, next_state, next_action, done);

        // Update the entire grid's greedy action/values
        for (int i = 0; i < kGridSize; i++)
        {
            for (int j = 0; j < kGridSize; j++)
            {
                q_values_grid[i][j] = agent.getMaxQValue({i, j}); // only for visualization
            }
        }

        state  = next_state;
        action = next_action;

        if (done)
        {
            resetState(state);
            done   = false;
            reward = 0.;

            // // decay the epsilon
            // if (agent.epsilon > 5. * DeepRlAgent::kEpsilonDecay)
            // {
            //     agent.epsilon -= DeepRlAgent::kEpsilonDecay;
            // }
            // else
            // {
            //     SetTargetFPS(10);
            // }
        }

        // Show all q values
        // showAllQValues(agent);

        std::cout << "total: " << ctr << " crash: " << crash_ctr << " goal_reach: " << goal_reach_ctr
                  << " eps: " << agent.epsilon << std::endl;
    }

    CloseWindow();
    return 0;
}
