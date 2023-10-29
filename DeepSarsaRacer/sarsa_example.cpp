#include "raylib.h"
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <vector>

const int gridSize     = 5;
const int cellSize     = 80;
const int screenWidth  = gridSize * cellSize;
const int screenHeight = gridSize * cellSize;

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
    int  x, y;
    bool operator==(const State &other) const
    {
        return x == other.x && y == other.y;
    }
};

namespace std
{
template <>
struct hash<State>
{
    std::size_t operator()(const State &k) const
    {
        return (k.x << 4) + k.y;
    }
};
} // namespace std

class SarsaAgent
{
  private:
    std::unordered_map<State, std::vector<double>> Q;
    double                                         alpha   = 0.1;
    double                                         gamma   = 0.9;
    double                                         epsilon = 0.2;

    Action greedyAction(const State &state)
    {
        if (Q[state].empty())
        {
            Q[state] = std::vector<double>(ACTION_COUNT, 0.0);
        }
        int bestIndex = 0;
        for (int i = 1; i < ACTION_COUNT; i++)
        {
            if (Q[state][i] > Q[state][bestIndex])
                bestIndex = i;
        }
        return static_cast<Action>(bestIndex);
    }

  public:
    Action chooseAction(const State &state)
    {
        if (Q[state].empty())
        {
            Q[state] = std::vector<double>(ACTION_COUNT, 0.0);
        }
        if (GetRandomValue(0, 99) < epsilon * 100)
        {
            return static_cast<Action>(GetRandomValue(0, ACTION_COUNT - 1));
        }
        return greedyAction(state);
    }

    void update(const State &oldState, Action action, double reward, const State &newState)
    {
        Action newAction    = greedyAction(newState);
        double oldQ         = Q[oldState][action];
        double maxNewQ      = Q[newState][newAction];
        Q[oldState][action] = oldQ + alpha * (reward + gamma * maxNewQ - oldQ);
    }

    std::vector<double> getQValues(const State &state)
    {
        if (Q[state].empty())
        {
            Q[state] = std::vector<double>(ACTION_COUNT, 0.0);
        }
        return Q[state];
    }
};

void drawStateValue(const State &state, const std::vector<double> &values)
{
    double stateValue = 0;
    for (double value : values)
    {
        stateValue += value;
    }

    char buffer[10];
    snprintf(buffer, sizeof(buffer), "%.2f", stateValue);

    Vector2 textPosition = {(state.x + 0.5) * cellSize - MeasureText(buffer, 20) / 2, (state.y + 0.5) * cellSize - 10};
    DrawText(buffer, textPosition.x, textPosition.y, 20, BLACK);
}

int main(void)
{
    InitWindow(screenWidth, screenHeight, "SARSA GridWorld");

    Vector2    goal          = {4, 4}; // Goal position
    State      agentPosition = {0, 0}; // Starting position
    SarsaAgent agent;

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Draw grid and state values
        for (int i = 0; i < gridSize; i++)
        {
            for (int j = 0; j < gridSize; j++)
            {
                DrawRectangleLines(i * cellSize, j * cellSize, cellSize, cellSize, LIGHTGRAY);
                drawStateValue({i, j}, agent.getQValues({i, j}));
            }
        }

        // Draw goal
        DrawRectangle(goal.x * cellSize, goal.y * cellSize, cellSize, cellSize, GREEN);

        // Draw agent
        DrawRectangle(agentPosition.x * cellSize, agentPosition.y * cellSize, cellSize, cellSize, BLUE);

        Action action = agent.chooseAction(agentPosition);

        State oldPosition = agentPosition;

        switch (action)
        {
        case UP:
            if (agentPosition.y > 0)
                agentPosition.y--;
            break;
        case DOWN:
            if (agentPosition.y < gridSize - 1)
                agentPosition.y++;
            break;
        case LEFT:
            if (agentPosition.x > 0)
                agentPosition.x--;
            break;
        case RIGHT:
            if (agentPosition.x < gridSize - 1)
                agentPosition.x++;
            break;
        }

        double reward = -0.04; // Small negative reward for each step
        if (agentPosition.x == goal.x && agentPosition.y == goal.y)
        {
            reward        = 1.0;
            agentPosition = {0, 0};
        }

        agent.update(oldPosition, action, reward, agentPosition);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
