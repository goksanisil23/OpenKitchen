#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Environment/Agent.h"
#include "NeatTypes.hpp"

namespace neat
{
inline bool isInputNode(const NodeGene &node)
{
    return node.type == NodeType::Input;
}

inline bool isBiasNode(const NodeGene &node)
{
    return node.type == NodeType::Bias;
}

inline bool isOutputNode(const NodeGene &node)
{
    return node.type == NodeType::Output;
}

inline bool isHiddenNode(const NodeGene &node)
{
    return node.type == NodeType::Hidden;
}

inline float randomFloat(std::mt19937 &random_engine, const float low, const float high)
{
    std::uniform_real_distribution<float> distribution(low, high);
    return distribution(random_engine);
}

inline bool sampleEvent(std::mt19937 &random_engine, const float probability)
{
    std::bernoulli_distribution distribution(probability);
    return distribution(random_engine);
}

template <typename T>
inline const T &randomSampleFromVector(const std::vector<T> &values, std::mt19937 &random_engine)
{
    std::uniform_int_distribution<size_t> distribution(0U, values.size() - 1U);
    return values[distribution(random_engine)];
}

inline std::vector<ConnectionGene> getConnectionsSortedByInnovation(const Genome &genome)
{
    std::vector<ConnectionGene> sorted_connections = genome.connections;
    std::sort(sorted_connections.begin(),
              sorted_connections.end(),
              [](const ConnectionGene &left_connection, const ConnectionGene &right_connection)
              { return left_connection.innovation_id < right_connection.innovation_id; });
    return sorted_connections;
}

inline const NodeGene &findNodeById(const Genome &genome, const int32_t node_id)
{
    const auto iter = std::find_if(
        genome.nodes.begin(), genome.nodes.end(), [node_id](const NodeGene &node) { return node.id == node_id; });
    require(iter != genome.nodes.end(), "Genome is missing a referenced node");
    return *iter;
}

inline bool genomeHasNode(const Genome &genome, const int32_t node_id)
{
    return std::any_of(
        genome.nodes.begin(), genome.nodes.end(), [node_id](const NodeGene &node) { return node.id == node_id; });
}

inline bool genomeHasConnection(const Genome &genome, const int32_t from_node_id, const int32_t to_node_id)
{
    return std::any_of(genome.connections.begin(),
                       genome.connections.end(),
                       [from_node_id, to_node_id](const ConnectionGene &connection)
                       { return connection.from_node_id == from_node_id && connection.to_node_id == to_node_id; });
}

inline float sigmoid(const float value)
{
    return 1.0F / (1.0F + std::exp(-Config::kSigmoidSlope * value));
}

// Kahn traversal keeps feed-forward evaluation valid and rejects accidental cycles.
inline std::vector<int32_t> buildTopologicalOrder(const Genome &genome)
{
    std::unordered_map<int32_t, std::vector<int32_t>> adjacency{};
    std::unordered_map<int32_t, int32_t>              indegree{};
    for (const auto &node : genome.nodes)
    {
        indegree.emplace(node.id, 0);
    }

    for (const auto &connection : genome.connections)
    {
        if (!connection.enabled)
        {
            continue;
        }
        adjacency[connection.from_node_id].push_back(connection.to_node_id);
        indegree.at(connection.to_node_id)++;
    }

    std::priority_queue<int32_t, std::vector<int32_t>, std::greater<int32_t>> ready_nodes{};
    for (const auto &[node_id, degree] : indegree)
    {
        if (degree == 0)
        {
            ready_nodes.push(node_id);
        }
    }

    std::vector<int32_t> ordered_nodes{};
    ordered_nodes.reserve(genome.nodes.size());
    while (!ready_nodes.empty())
    {
        const int32_t node_id = ready_nodes.top();
        ready_nodes.pop();
        ordered_nodes.push_back(node_id);

        const auto adjacency_iter = adjacency.find(node_id);
        if (adjacency_iter == adjacency.end())
        {
            continue;
        }

        for (const int32_t next_node_id : adjacency_iter->second)
        {
            indegree.at(next_node_id)--;
            if (indegree.at(next_node_id) == 0)
            {
                ready_nodes.push(next_node_id);
            }
        }
    }

    require(ordered_nodes.size() == genome.nodes.size(), "Genome contains a cycle");

    return ordered_nodes;
}

// Evaluate the current DAG genome directly from normalized ray distances plus the internal bias node.
inline std::array<float, Config::kNumOutputs> evaluateGenome(const Genome             &genome,
                                                             const std::vector<Vec2d> &sensor_hits)
{
    require(static_cast<int32_t>(sensor_hits.size()) == Config::kNumRayInputs,
            "Unexpected number of raycast inputs for NEAT evaluation");

    const std::vector<int32_t>                                       topological_order = buildTopologicalOrder(genome);
    std::unordered_map<int32_t, std::vector<const ConnectionGene *>> incoming_connections{};
    for (const auto &connection : genome.connections)
    {
        if (connection.enabled)
        {
            incoming_connections[connection.to_node_id].push_back(&connection);
        }
    }

    std::unordered_map<int32_t, float> node_values{};
    node_values.reserve(genome.nodes.size());
    for (const auto &node : genome.nodes)
    {
        if (isInputNode(node))
        {
            const int32_t ray_index = node.id;
            node_values[node.id]    = sensor_hits.at(ray_index).norm() / Agent::kSensorRange;
        }
        else if (isBiasNode(node))
        {
            node_values[node.id] = 1.0F;
        }
    }

    for (const int32_t node_id : topological_order)
    {
        const NodeGene &node = findNodeById(genome, node_id);
        if (isInputNode(node) || isBiasNode(node))
        {
            continue;
        }

        float      sum           = 0.0F;
        const auto incoming_iter = incoming_connections.find(node_id);
        if (incoming_iter != incoming_connections.end())
        {
            for (const ConnectionGene *connection : incoming_iter->second)
            {
                sum += node_values.at(connection->from_node_id) * connection->weight;
            }
        }

        node_values[node_id] = sigmoid(sum);
    }

    std::vector<int32_t> output_node_ids{};
    output_node_ids.reserve(Config::kNumOutputs);
    for (const auto &node : genome.nodes)
    {
        if (isOutputNode(node))
        {
            output_node_ids.push_back(node.id);
        }
    }
    std::sort(output_node_ids.begin(), output_node_ids.end());

    require(static_cast<int32_t>(output_node_ids.size()) == Config::kNumOutputs,
            "Genome output count does not match configuration");

    std::array<float, Config::kNumOutputs> outputs{};
    for (int32_t output_idx = 0; output_idx < Config::kNumOutputs; ++output_idx)
    {
        outputs.at(output_idx) = node_values.at(output_node_ids.at(output_idx));
    }
    return outputs;
}

// Vanilla NEAT speciation metric over excess, disjoint, and matching-weight differences.
inline float compatibilityDistance(const Genome &left_genome, const Genome &right_genome)
{
    const std::vector<ConnectionGene> left_connections  = getConnectionsSortedByInnovation(left_genome);
    const std::vector<ConnectionGene> right_connections = getConnectionsSortedByInnovation(right_genome);

    size_t matching_gene_count = 0U;
    size_t disjoint_gene_count = 0U;
    size_t excess_gene_count   = 0U;
    float  weight_diff_sum     = 0.0F;

    size_t left_index{0U};
    size_t right_index{0U};
    while (left_index < left_connections.size() && right_index < right_connections.size())
    {
        const int32_t left_innovation_id  = left_connections[left_index].innovation_id;
        const int32_t right_innovation_id = right_connections[right_index].innovation_id;
        if (left_innovation_id == right_innovation_id)
        {
            matching_gene_count++;
            weight_diff_sum += std::fabs(left_connections[left_index].weight - right_connections[right_index].weight);
            left_index++;
            right_index++;
        }
        else if (left_innovation_id < right_innovation_id)
        {
            disjoint_gene_count++;
            left_index++;
        }
        else
        {
            disjoint_gene_count++;
            right_index++;
        }
    }

    excess_gene_count += left_connections.size() - left_index;
    excess_gene_count += right_connections.size() - right_index;

    const float normalizer = (std::max(left_connections.size(), right_connections.size()) < 20U)
                                 ? 1.0F
                                 : static_cast<float>(std::max(left_connections.size(), right_connections.size()));
    const float average_weight_diff = (matching_gene_count == 0U) ? 0.0F : (weight_diff_sum / matching_gene_count);

    return (Config::kExcessCoeff * static_cast<float>(excess_gene_count) / normalizer) +
           (Config::kDisjointCoeff * static_cast<float>(disjoint_gene_count) / normalizer) +
           (Config::kWeightDiffCoeff * average_weight_diff);
}

// Reject new edges that would make the feed-forward genome cyclic.
inline bool wouldIntroduceCycle(const Genome &genome, const int32_t from_node_id, const int32_t to_node_id)
{
    if (from_node_id == to_node_id)
    {
        return true;
    }

    std::unordered_map<int32_t, std::vector<int32_t>> adjacency{};
    for (const auto &connection : genome.connections)
    {
        if (connection.enabled)
        {
            adjacency[connection.from_node_id].push_back(connection.to_node_id);
        }
    }

    std::vector<int32_t>        frontier{to_node_id};
    std::unordered_set<int32_t> visited{};
    while (!frontier.empty())
    {
        const int32_t node_id = frontier.back();
        frontier.pop_back();
        if (node_id == from_node_id)
        {
            return true;
        }
        if (!visited.insert(node_id).second)
        {
            continue;
        }

        const auto adjacency_iter = adjacency.find(node_id);
        if (adjacency_iter == adjacency.end())
        {
            continue;
        }
        frontier.insert(frontier.end(), adjacency_iter->second.begin(), adjacency_iter->second.end());
    }

    return false;
}

inline Genome createMinimalGenome(InnovationTracker &innovation_tracker, std::mt19937 &random_engine)
{
    Genome genome{};
    genome.nodes.reserve(Config::kNumRayInputs + 1 + Config::kNumOutputs);
    for (int32_t input_idx = 0; input_idx < Config::kNumRayInputs; ++input_idx)
    {
        genome.nodes.push_back(NodeGene{input_idx, NodeType::Input});
    }

    const int32_t bias_node_id = Config::kNumRayInputs;
    genome.nodes.push_back(NodeGene{bias_node_id, NodeType::Bias});

    const int32_t output_node_offset = Config::kNumRayInputs + 1;
    for (int32_t output_idx = 0; output_idx < Config::kNumOutputs; ++output_idx)
    {
        genome.nodes.push_back(NodeGene{output_node_offset + output_idx, NodeType::Output});
    }

    for (int32_t source_node_id = 0; source_node_id <= bias_node_id; ++source_node_id)
    {
        for (int32_t output_idx = 0; output_idx < Config::kNumOutputs; ++output_idx)
        {
            const int32_t target_node_id = output_node_offset + output_idx;
            genome.connections.push_back(
                ConnectionGene{innovation_tracker.getConnectionInnovationId(source_node_id, target_node_id),
                               source_node_id,
                               target_node_id,
                               randomFloat(random_engine, Config::kWeightInitMin, Config::kWeightInitMax),
                               true});
        }
    }

    return genome;
}

inline void mutateWeights(Genome &genome, std::mt19937 &random_engine)
{
    for (auto &connection : genome.connections)
    {
        if (!sampleEvent(random_engine, Config::kWeightMutationRate))
        {
            continue;
        }

        if (sampleEvent(random_engine, Config::kWeightPerturbRate))
        {
            connection.weight += randomFloat(random_engine, -Config::kWeightPerturbScale, Config::kWeightPerturbScale);
        }
        else
        {
            connection.weight = randomFloat(random_engine, Config::kWeightInitMin, Config::kWeightInitMax);
        }
    }
}

// Sample a new valid feed-forward edge from the currently reachable node pairs.
inline void mutateAddConnection(Genome &genome, InnovationTracker &innovation_tracker, std::mt19937 &random_engine)
{
    if (!sampleEvent(random_engine, Config::kAddConnectionRate))
    {
        return;
    }

    std::vector<std::pair<int32_t, int32_t>> candidate_connections{};
    for (const auto &from_node : genome.nodes)
    {
        if (isOutputNode(from_node))
        {
            continue;
        }

        for (const auto &to_node : genome.nodes)
        {
            if (isInputNode(to_node) || isBiasNode(to_node))
            {
                continue;
            }
            if (genomeHasConnection(genome, from_node.id, to_node.id))
            {
                continue;
            }
            if (wouldIntroduceCycle(genome, from_node.id, to_node.id))
            {
                continue;
            }
            candidate_connections.emplace_back(from_node.id, to_node.id);
        }
    }

    if (candidate_connections.empty())
    {
        return;
    }

    const auto &[from_node_id, to_node_id] = randomSampleFromVector(candidate_connections, random_engine);
    genome.connections.push_back(
        ConnectionGene{innovation_tracker.getConnectionInnovationId(from_node_id, to_node_id),
                       from_node_id,
                       to_node_id,
                       randomFloat(random_engine, Config::kWeightInitMin, Config::kWeightInitMax),
                       true});
}

// Split one enabled edge into two edges with a new hidden node, preserving NEAT innovation history.
inline void mutateAddNode(Genome &genome, InnovationTracker &innovation_tracker, std::mt19937 &random_engine)
{
    if (!sampleEvent(random_engine, Config::kAddNodeRate))
    {
        return;
    }

    std::vector<size_t> enabled_connection_indices{};
    for (size_t connection_idx = 0U; connection_idx < genome.connections.size(); ++connection_idx)
    {
        if (genome.connections[connection_idx].enabled)
        {
            enabled_connection_indices.push_back(connection_idx);
        }
    }

    if (enabled_connection_indices.empty())
    {
        return;
    }

    const size_t         selected_index        = randomSampleFromVector(enabled_connection_indices, random_engine);
    const ConnectionGene split_connection      = genome.connections[selected_index];
    genome.connections[selected_index].enabled = false;

    const InnovationTracker::SplitRecord split_record = innovation_tracker.getOrCreateSplitRecord(split_connection);
    if (!genomeHasNode(genome, split_record.new_node_id))
    {
        genome.nodes.push_back(NodeGene{split_record.new_node_id, NodeType::Hidden});
    }

    if (!genomeHasConnection(genome, split_connection.from_node_id, split_record.new_node_id))
    {
        genome.connections.push_back(ConnectionGene{
            split_record.inbound_innovation_id, split_connection.from_node_id, split_record.new_node_id, 1.0F, true});
    }

    if (!genomeHasConnection(genome, split_record.new_node_id, split_connection.to_node_id))
    {
        genome.connections.push_back(ConnectionGene{split_record.outbound_innovation_id,
                                                    split_record.new_node_id,
                                                    split_connection.to_node_id,
                                                    split_connection.weight,
                                                    true});
    }
}

inline void mutateReenableConnection(Genome &genome, std::mt19937 &random_engine)
{
    if (!sampleEvent(random_engine, Config::kReenableConnectionRate))
    {
        return;
    }

    std::vector<size_t> disabled_connection_indices{};
    for (size_t connection_idx = 0U; connection_idx < genome.connections.size(); ++connection_idx)
    {
        if (!genome.connections[connection_idx].enabled)
        {
            disabled_connection_indices.push_back(connection_idx);
        }
    }

    if (disabled_connection_indices.empty())
    {
        return;
    }

    const size_t selected_index                = randomSampleFromVector(disabled_connection_indices, random_engine);
    genome.connections[selected_index].enabled = true;
}

inline void mutateGenome(Genome &genome, InnovationTracker &innovation_tracker, std::mt19937 &random_engine)
{
    mutateWeights(genome, random_engine);
    mutateAddConnection(genome, innovation_tracker, random_engine);
    mutateAddNode(genome, innovation_tracker, random_engine);
    mutateReenableConnection(genome, random_engine);
}

// Merge aligned genes by innovation id and keep non-matching genes from the fitter parent.
inline Genome crossoverGenomes(const Genome &left_parent, const Genome &right_parent, std::mt19937 &random_engine)
{
    const bool left_is_fitter  = left_parent.raw_fitness > right_parent.raw_fitness;
    const bool right_is_fitter = right_parent.raw_fitness > left_parent.raw_fitness;
    const bool equal_fitness   = !left_is_fitter && !right_is_fitter;

    const Genome &fitter_parent = left_is_fitter ? left_parent : right_parent;

    const std::vector<ConnectionGene> left_connections  = getConnectionsSortedByInnovation(left_parent);
    const std::vector<ConnectionGene> right_connections = getConnectionsSortedByInnovation(right_parent);

    std::unordered_map<int32_t, NodeGene> all_nodes{};
    for (const auto &node : left_parent.nodes)
    {
        all_nodes.emplace(node.id, node);
    }
    for (const auto &node : right_parent.nodes)
    {
        all_nodes.emplace(node.id, node);
    }

    Genome child{};
    size_t left_index{0U};
    size_t right_index{0U};
    while (left_index < left_connections.size() || right_index < right_connections.size())
    {
        if (left_index >= left_connections.size())
        {
            if (equal_fitness || right_is_fitter)
            {
                child.connections.push_back(right_connections[right_index]);
            }
            right_index++;
            continue;
        }

        if (right_index >= right_connections.size())
        {
            if (equal_fitness || left_is_fitter)
            {
                child.connections.push_back(left_connections[left_index]);
            }
            left_index++;
            continue;
        }

        const ConnectionGene &left_connection  = left_connections[left_index];
        const ConnectionGene &right_connection = right_connections[right_index];
        if (left_connection.innovation_id == right_connection.innovation_id)
        {
            ConnectionGene inherited_connection = sampleEvent(random_engine, 0.5F) ? left_connection : right_connection;
            if (!left_connection.enabled || !right_connection.enabled)
            {
                inherited_connection.enabled = !sampleEvent(random_engine, Config::kDisabledInheritanceRate);
            }
            child.connections.push_back(inherited_connection);
            left_index++;
            right_index++;
            continue;
        }

        if (left_connection.innovation_id < right_connection.innovation_id)
        {
            if (equal_fitness || left_is_fitter)
            {
                child.connections.push_back(left_connection);
            }
            left_index++;
            continue;
        }

        if (equal_fitness || right_is_fitter)
        {
            child.connections.push_back(right_connection);
        }
        right_index++;
    }

    std::unordered_set<int32_t> referenced_node_ids{};
    for (const auto &connection : child.connections)
    {
        referenced_node_ids.insert(connection.from_node_id);
        referenced_node_ids.insert(connection.to_node_id);
    }

    for (const auto &[node_id, node] : all_nodes)
    {
        if (!isHiddenNode(node) || referenced_node_ids.count(node_id) > 0U)
        {
            child.nodes.push_back(node);
        }
    }

    std::sort(child.nodes.begin(),
              child.nodes.end(),
              [](const NodeGene &left_node, const NodeGene &right_node) { return left_node.id < right_node.id; });
    std::sort(child.connections.begin(),
              child.connections.end(),
              [](const ConnectionGene &left_connection, const ConnectionGene &right_connection)
              { return left_connection.innovation_id < right_connection.innovation_id; });

    if (child.nodes.empty())
    {
        child = fitter_parent;
    }

    return child;
}
} // namespace neat
