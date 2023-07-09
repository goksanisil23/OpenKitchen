#include "raylib-cpp.hpp"

#include <iostream>
#include <queue>
#include <stack>
#include <unordered_set>
#include <vector>

struct Node
{
    int             id{-1};
    raylib::Vector2 position;
    raylib::Color   color{raylib::Color::Green()};
};

void topoSortDfs(const std::vector<std::vector<int>> &adj_list,
                 const int                            node_id,
                 std::unordered_set<int>             &visited,
                 std::vector<int>                    &topo_order)
{
    if (visited.find(node_id) != visited.end())
    {
        return;
    }
    for (const int child_id : adj_list[node_id])
    {
        topoSortDfs(adj_list, child_id, visited, topo_order);
    }

    topo_order.push_back(node_id);
    visited.insert(node_id);
}

std::vector<int> topologicalSortRecursive(const std::vector<std::vector<int>> &adj_list)
{
    std::vector<int>        topo_order{};
    std::unordered_set<int> visited;

    for (int node_id = 0; node_id < static_cast<int>(adj_list.size()); node_id++)
    {
        topoSortDfs(adj_list, node_id, visited, topo_order);
    }

    return topo_order;
}

// In graphs with multiple root nodes, it cannot sort the root nodes correctly, because as soon as
// initial root node is popped, we continue towards it's children, instead of treating the other root nodes as well.
std::vector<int> topologicalSortIterative(const std::vector<std::vector<int>> &adj_list)
{
    std::vector<int> topo_order{};
    std::vector<int> indegrees(adj_list.size()); // counting indegrees of each node
    // std::stack<int>  nodes_with_no_incoming_edges;
    std::vector<int> nodes_with_no_incoming_edges;

    for (int node_id{0}; node_id < static_cast<int>(adj_list.size()); node_id++)
    {
        for (int child_id : adj_list[node_id])
        {
            indegrees[child_id] += 1;
        }
    }

    for (int node_id{0}; node_id < static_cast<int>(adj_list.size()); node_id++)
    {
        if (indegrees[node_id] == 0)
        {
            std::cout << "nodes_with_no_incoming_edges: " << node_id << std::endl;
            nodes_with_no_incoming_edges.push_back(node_id);
        }
    }

    while (nodes_with_no_incoming_edges.size() > 0)
    {
        // const int node = nodes_with_no_incoming_edges.top();
        const int node = nodes_with_no_incoming_edges.back();
        std::cout << "popping: " << node << std::endl;
        nodes_with_no_incoming_edges.pop_back();
        topo_order.push_back(node);

        // decrement the indegree of that node's neighbors
        for (const auto child_id : adj_list[node])
        {
            indegrees[child_id] -= 1;
            if (indegrees[child_id] == 0)
            {
                nodes_with_no_incoming_edges.push_back(child_id);
            }
        }
    }

    return topo_order;
}

void drawArrow(raylib::Vector2 start, raylib::Vector2 end)
{
    // Calculate the arrow direction and length
    raylib::Vector2 direction = end - start;
    float           length    = direction.Length();

    // Calculate the normalized arrow direction
    raylib::Vector2 normalizedDirection = direction.Normalize();

    // Calculate the arrowhead size
    const float arrowheadSize = 20.0f;

    // Calculate the position of the arrowhead
    raylib::Vector2 arrowheadPos = start + normalizedDirection * (length - arrowheadSize);

    // Draw the arrow body (line)
    DrawLineV(start, end, raylib::Color::Yellow());

    // Draw the arrowhead (triangle)
    DrawTriangle(arrowheadPos,
                 arrowheadPos + (-normalizedDirection.Rotate(45)) * arrowheadSize,
                 arrowheadPos + (-normalizedDirection.Rotate(-45)) * arrowheadSize,
                 raylib::Color::Yellow());
}

void visualizeSortedGraph(const std::vector<std::vector<int>> &adj_list, const std::vector<int> &topo_list)
{
    constexpr int   kScreenWidth  = 800;
    constexpr int   kScreenHeight = 600;
    constexpr float kRootNodePosX{kScreenWidth / 2.0};
    constexpr float kRootNodePosY{kScreenHeight * 0.2};

    // // Create visualization nodes
    std::vector<Node> nodes(topo_list.size());
    int               i = 0;
    for (auto node_itr = topo_list.rbegin(); node_itr != topo_list.rend(); node_itr++)
    {
        float lateral_offset           = ((i % 2) == 0) ? 100 : -100; // alternate between left and right
        nodes.at(*node_itr).id         = *node_itr;
        nodes.at(*node_itr).position.x = kRootNodePosX + lateral_offset;
        nodes.at(*node_itr).position.y = kRootNodePosY + i * 70;
        nodes.at(*node_itr).color      = RED;

        i++;
    }

    raylib::Window window(kScreenWidth, kScreenHeight, "Graph Visualization");
    window.SetTargetFPS(60);

    while (!window.ShouldClose())
    {
        window.BeginDrawing();
        window.ClearBackground(BLACK);

        for (auto node_itr = topo_list.rbegin(); node_itr != topo_list.rend(); node_itr++)
        {
            DrawCircleV(nodes.at(*node_itr).position, 20, nodes.at(*node_itr).color);

            // Draw the edges towards children
            for (const int child_id : adj_list[*node_itr])
            {
                raylib::Vector2 startPos = nodes.at(*node_itr).position;
                raylib::Vector2 endPos   = nodes.at(child_id).position;
                // DrawLineEx(startPos, endPos, 3, YELLOW);
                drawArrow(startPos, endPos);
            }

            DrawText(std::to_string(nodes.at(*node_itr).id),
                     nodes.at(*node_itr).position.GetX() - 10,
                     nodes.at(*node_itr).position.GetY() - 10,
                     30,
                     raylib::Color::Blue());
        }
        window.EndDrawing();
    }

    window.Close();
}

int main()
{

    constexpr int kNumNodes1 = 6;
    constexpr int kNumNodes2 = 5;

    // Adjacency list representation of the graph: outer = parent, inner = children
    std::vector<std::vector<int>> adj_list_1(kNumNodes1);
    std::vector<std::vector<int>> adj_list_2(kNumNodes2);

    // Adding directed edges to the graph
    adj_list_1[5].push_back(2);
    adj_list_1[5].push_back(0);
    adj_list_1[4].push_back(0);
    adj_list_1[4].push_back(1);
    adj_list_1[2].push_back(3);
    adj_list_1[3].push_back(1);

    adj_list_2[0].push_back(1);
    adj_list_2[0].push_back(2);
    adj_list_2[1].push_back(3);
    adj_list_2[2].push_back(3);
    adj_list_2[1].push_back(4);
    adj_list_2[3].push_back(4);

    // Perform topological sort
    auto topo_list_1{topologicalSortRecursive(adj_list_1)};
    auto topo_list_2{topologicalSortRecursive(adj_list_2)};
    // auto topo_list{topologicalSortIterative(adj_list)};

    visualizeSortedGraph(adj_list_1, topo_list_1);
    visualizeSortedGraph(adj_list_2, topo_list_2);

    return 0;
}
