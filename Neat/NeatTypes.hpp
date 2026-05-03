#pragma once

#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NeatConfig.hpp"

namespace neat
{
inline void require(const bool condition, const char *const message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

enum class NodeType
{
    Input = 0,
    Bias,
    Hidden,
    Output
};

struct NodeGene
{
    int32_t  id{};
    NodeType type{NodeType::Hidden};
};

struct ConnectionGene
{
    int32_t innovation_id{};
    int32_t from_node_id{};
    int32_t to_node_id{};
    float   weight{};
    bool    enabled{true};
};

struct Genome
{
    std::vector<NodeGene>       nodes{};
    std::vector<ConnectionGene> connections{};
    float                       raw_fitness{0.0F};
    float                       adjusted_fitness{0.0F};
    int32_t                     species_id{-1};
};

inline void resetGenomeEvaluation(Genome &genome, const float raw_fitness = 0.0F)
{
    genome.raw_fitness      = raw_fitness;
    genome.adjusted_fitness = 0.0F;
    genome.species_id       = -1;
}

struct Species
{
    int32_t             id{-1};
    Genome              representative{};
    std::vector<size_t> member_genome_indices{};
    float               best_raw_fitness{-std::numeric_limits<float>::infinity()};
    uint32_t            stagnation_counter{0U};
    float               total_adjusted_fitness{0.0F};
};

struct GenerationStats
{
    float   best_fitness{0.0F};
    float   average_fitness{0.0F};
    int32_t species_count{0};
};

struct FitnessTracker
{
    bool    initialized{false};
    int32_t previous_track_idx{0};
    float   cumulative_progress{0.0F};
    float   best_progress{0.0F};
};

struct ConnectionKeyHash
{
    size_t operator()(const std::pair<int32_t, int32_t> &key) const
    {
        const size_t first_hash  = std::hash<int32_t>{}(key.first);
        const size_t second_hash = std::hash<int32_t>{}(key.second);
        return first_hash ^ (second_hash << 1U);
    }
};

class InnovationTracker
{
  public:
    struct SplitRecord
    {
        int32_t new_node_id{};
        int32_t inbound_innovation_id{};
        int32_t outbound_innovation_id{};
    };

    InnovationTracker() : next_node_id_{Config::kNumRayInputs + 1 + Config::kNumOutputs}
    {
    }

    int32_t getConnectionInnovationId(const int32_t from_node_id, const int32_t to_node_id)
    {
        const std::pair<int32_t, int32_t> key{from_node_id, to_node_id};
        const auto                        iter = connection_innovations_.find(key);
        if (iter != connection_innovations_.end())
        {
            return iter->second;
        }

        const int32_t innovation_id = next_innovation_id_++;
        connection_innovations_.emplace(key, innovation_id);
        return innovation_id;
    }

    // Reuse the same split history so structurally identical node mutations line up across genomes.
    SplitRecord getOrCreateSplitRecord(const ConnectionGene &connection)
    {
        const auto iter = split_history_.find(connection.innovation_id);
        if (iter != split_history_.end())
        {
            return iter->second;
        }

        const int32_t     new_node_id = next_node_id_++;
        const SplitRecord split_record{new_node_id,
                                       getConnectionInnovationId(connection.from_node_id, new_node_id),
                                       getConnectionInnovationId(new_node_id, connection.to_node_id)};
        split_history_.emplace(connection.innovation_id, split_record);
        return split_record;
    }

  private:
    int32_t                                                                     next_node_id_{0};
    int32_t                                                                     next_innovation_id_{0};
    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, ConnectionKeyHash> connection_innovations_{};
    std::unordered_map<int32_t, SplitRecord>                                    split_history_{};
};
} // namespace neat
