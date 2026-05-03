#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "Environment/RaceTrack.h"
#include "NeatAgent.hpp"

namespace neat
{
class Population
{
  public:
    Population(const uint16_t population_size) : random_engine_{Config::kRandomSeed}
    {
        genomes_.reserve(population_size);
        fitness_trackers_.resize(population_size);
        for (int16_t genome_idx = 0; genome_idx < population_size; ++genome_idx)
        {
            genomes_.push_back(createMinimalGenome(innovation_tracker_, random_engine_));
        }
    }

    void setAgents(const std::vector<std::unique_ptr<NeatAgent>> &agents)
    {
        require(agents.size() == genomes_.size(), "Population size does not match agent count");

        for (size_t genome_idx = 0U; genome_idx < genomes_.size(); ++genome_idx)
        {
            agents[genome_idx]->bindGenome(&genomes_[genome_idx]);
        }
    }

    void resetFitnessTracking()
    {
        std::fill(fitness_trackers_.begin(), fitness_trackers_.end(), FitnessTracker{});
    }

    // Track unwrapped lap progress so crossing the start line increases fitness instead of resetting it.
    void updateFitnessTracking(const std::vector<std::unique_ptr<NeatAgent>> &agents, const RaceTrack &race_track)
    {
        require(agents.size() == genomes_.size(), "Population size does not match agent count");

        const int32_t track_length = static_cast<int32_t>(race_track.track_data_points_.x_m.size());

        for (size_t genome_idx = 0U; genome_idx < genomes_.size(); ++genome_idx)
        {
            auto         &fitness_tracker = fitness_trackers_[genome_idx];
            const int32_t nearest_track_idx =
                static_cast<int32_t>(race_track.findNearestTrackIndexBruteForce(agents[genome_idx]->pos_));

            if (!fitness_tracker.initialized)
            {
                fitness_tracker.initialized         = true;
                fitness_tracker.previous_track_idx  = nearest_track_idx;
                fitness_tracker.cumulative_progress = static_cast<float>(nearest_track_idx);
                fitness_tracker.best_progress       = fitness_tracker.cumulative_progress;
                continue;
            }

            int32_t       delta_idx      = nearest_track_idx - fitness_tracker.previous_track_idx;
            const int32_t wrap_threshold = track_length / 2;
            if (delta_idx < -wrap_threshold)
            {
                delta_idx += track_length;
            }
            else if (delta_idx > wrap_threshold)
            {
                delta_idx -= track_length;
            }

            fitness_tracker.cumulative_progress += static_cast<float>(delta_idx);
            fitness_tracker.best_progress =
                std::max(fitness_tracker.best_progress, fitness_tracker.cumulative_progress);
            fitness_tracker.previous_track_idx = nearest_track_idx;
        }
    }

    void assignFitnessFromTracking()
    {
        require(fitness_trackers_.size() == genomes_.size(), "Population size does not match fitness tracker count");

        for (size_t genome_idx = 0U; genome_idx < genomes_.size(); ++genome_idx)
        {
            resetGenomeEvaluation(genomes_[genome_idx], fitness_trackers_[genome_idx].best_progress);
        }
    }

    // Rebuild species assignments around current representatives and refresh shared adjusted fitness.
    // 1) Start from previous generation's species representatives to assign current genomes into species
    // 2) After assignment, replace the representative of each species with a random genome in that species
    void speciate()
    {
        std::vector<Species> next_group_of_species = group_of_species_;
        // Reset who belongs to which species, but keep the representatives so species identities persist across generations.
        for (auto &species : next_group_of_species)
        {
            species.member_genome_indices.clear();
            species.total_adjusted_fitness = 0.0F;
        }

        for (size_t genome_idx = 0U; genome_idx < genomes_.size(); ++genome_idx)
        {
            bool assigned = false;
            for (auto &species : next_group_of_species)
            {
                const auto &representative_genome  = species.representative;
                const float dist_to_representative = compatibilityDistance(genomes_[genome_idx], representative_genome);
                if (dist_to_representative <= Config::kCompatibilityThreshold)
                {
                    species.member_genome_indices.push_back(genome_idx);
                    genomes_[genome_idx].species_id = species.id;
                    assigned                        = true;
                    break;
                }
            }

            // No existing species was close enough, so create a new one with this genome as representative.
            if (!assigned)
            {
                Species new_species{};
                new_species.id             = next_group_of_species_id_++;
                new_species.representative = genomes_[genome_idx];
                new_species.member_genome_indices.push_back(genome_idx);
                genomes_[genome_idx].species_id = new_species.id;
                next_group_of_species.push_back(new_species);
            }
        }

        std::vector<Species> filtered_group_of_species{};
        filtered_group_of_species.reserve(next_group_of_species.size());
        for (auto &species : next_group_of_species)
        {
            if (species.member_genome_indices.empty())
            {
                continue;
            }

            species.representative = genomes_[randomSampleFromVector(species.member_genome_indices, random_engine_)];
            float best_fitness_in_species{-std::numeric_limits<float>::infinity()};
            for (const size_t genome_idx : species.member_genome_indices)
            {
                const float adjusted_fitness =
                    genomes_[genome_idx].raw_fitness / static_cast<float>(species.member_genome_indices.size());
                genomes_[genome_idx].adjusted_fitness = adjusted_fitness;
                species.total_adjusted_fitness += adjusted_fitness;
                best_fitness_in_species = std::max(best_fitness_in_species, genomes_[genome_idx].raw_fitness);
            }

            if (best_fitness_in_species > species.best_raw_fitness)
            {
                species.best_raw_fitness   = best_fitness_in_species;
                species.stagnation_counter = 0U;
            }
            else
            {
                species.stagnation_counter++;
            }

            filtered_group_of_species.push_back(species);
        }

        group_of_species_ = std::move(filtered_group_of_species);
    }

    GenerationStats getGenerationStats() const
    {
        GenerationStats stats{};
        stats.species_count = static_cast<int32_t>(group_of_species_.size());

        if (genomes_.empty())
        {
            return stats;
        }

        float fitness_sum{0.0F};
        stats.best_fitness = genomes_.front().raw_fitness;
        for (const auto &genome : genomes_)
        {
            stats.best_fitness = std::max(stats.best_fitness, genome.raw_fitness);
            fitness_sum += genome.raw_fitness;
        }
        stats.average_fitness = fitness_sum / static_cast<float>(genomes_.size());
        return stats;
    }

    // Build the next generation with elitism, shared-fitness offspring allocation, crossover, and mutation.
    void evolve()
    {
        require(!genomes_.empty(), "Population is empty");

        const size_t champion_index = static_cast<size_t>(
            std::distance(genomes_.begin(),
                          std::max_element(genomes_.begin(),
                                           genomes_.end(),
                                           [](const Genome &left_genome, const Genome &right_genome)
                                           { return left_genome.raw_fitness < right_genome.raw_fitness; })));
        const Genome  elite_genome     = genomes_[champion_index];
        const int32_t champion_species = elite_genome.species_id;

        std::vector<const Species *> active_species{};
        for (const auto &species : group_of_species_)
        {
            const bool keep_species =
                (species.stagnation_counter < Config::kStagnationGenerations) || (species.id == champion_species);
            if (keep_species && !species.member_genome_indices.empty())
            {
                active_species.push_back(&species);
            }
        }

        std::vector<Genome> next_generation{};
        next_generation.reserve(genomes_.size());
        next_generation.push_back(elite_genome);

        if (active_species.empty())
        {
            while (next_generation.size() < genomes_.size())
            {
                appendMutatedGenome(next_generation, Genome{elite_genome});
            }
            genomes_ = std::move(next_generation);
            generation_index_++;
            return;
        }

        const size_t       remaining_slots = genomes_.size() - 1U;
        std::vector<float> species_weights{};
        species_weights.reserve(active_species.size());
        float total_weight{0.0F};
        for (const Species *species : active_species)
        {
            const float weight = std::max(species->total_adjusted_fitness, 0.0F);
            species_weights.push_back(weight);
            total_weight += weight;
        }
        if (total_weight == 0.0F)
        {
            std::fill(
                species_weights.begin(), species_weights.end(), 1.0F / static_cast<float>(species_weights.size()));
        }

        std::vector<size_t> offspring_counts(active_species.size(), 0U);
        std::vector<float>  remainders(active_species.size(), 0.0F);
        size_t              allocated_slots{0U};
        const float normalized_weight_sum = std::accumulate(species_weights.begin(), species_weights.end(), 0.0F);
        for (size_t species_idx = 0U; species_idx < active_species.size(); ++species_idx)
        {
            const float normalized_weight = species_weights[species_idx] / normalized_weight_sum;
            const float exact_offspring   = normalized_weight * static_cast<float>(remaining_slots);
            offspring_counts[species_idx] = static_cast<size_t>(std::floor(exact_offspring));
            remainders[species_idx]       = exact_offspring - static_cast<float>(offspring_counts[species_idx]);
            allocated_slots += offspring_counts[species_idx];
        }

        while (allocated_slots < remaining_slots)
        {
            const size_t best_fractional_idx = static_cast<size_t>(
                std::distance(remainders.begin(), std::max_element(remainders.begin(), remainders.end())));
            offspring_counts[best_fractional_idx]++;
            remainders[best_fractional_idx] = 0.0F;
            allocated_slots++;
        }

        for (size_t species_idx = 0U; species_idx < active_species.size(); ++species_idx)
        {
            const Species &species = *active_species[species_idx];
            if (offspring_counts[species_idx] == 0U)
            {
                continue;
            }

            std::vector<size_t> ranked_members = species.member_genome_indices;
            std::sort(ranked_members.begin(),
                      ranked_members.end(),
                      [this](const size_t left_idx, const size_t right_idx)
                      { return genomes_[left_idx].raw_fitness > genomes_[right_idx].raw_fitness; });

            const size_t survivor_count = std::max<size_t>(
                1U,
                static_cast<size_t>(std::ceil(static_cast<float>(ranked_members.size()) * Config::kSurvivalFraction)));
            ranked_members.resize(survivor_count);

            for (size_t child_idx = 0U; child_idx < offspring_counts[species_idx]; ++child_idx)
            {
                Genome child{};
                if ((ranked_members.size() == 1U) || !sampleEvent(random_engine_, Config::kCrossoverRate))
                {
                    child = genomes_[selectParentIndex(ranked_members)];
                }
                else
                {
                    const size_t first_parent_idx  = selectParentIndex(ranked_members);
                    size_t       second_parent_idx = selectParentIndex(ranked_members);
                    if (ranked_members.size() > 1U)
                    {
                        while (second_parent_idx == first_parent_idx)
                        {
                            second_parent_idx = selectParentIndex(ranked_members);
                        }
                    }

                    child = crossoverGenomes(genomes_[first_parent_idx], genomes_[second_parent_idx], random_engine_);
                }

                appendMutatedGenome(next_generation, std::move(child));
            }
        }

        while (next_generation.size() < genomes_.size())
        {
            appendMutatedGenome(next_generation, Genome{elite_genome});
        }

        if (next_generation.size() > genomes_.size())
        {
            next_generation.resize(genomes_.size());
        }

        genomes_ = std::move(next_generation);
        generation_index_++;
    }

  private:
    // Centralize post-reproduction mutation and evaluation-state reset before storing offspring.
    void appendMutatedGenome(std::vector<Genome> &next_generation, Genome genome)
    {
        mutateGenome(genome, innovation_tracker_, random_engine_);
        resetGenomeEvaluation(genome);
        next_generation.push_back(std::move(genome));
    }

    // Use raw fitness as the within-species parent selection weight, with uniform fallback for all-zero fitness.
    size_t selectParentIndex(const std::vector<size_t> &member_indices)
    {
        std::vector<float> weights{};
        weights.reserve(member_indices.size());
        for (const size_t member_idx : member_indices)
        {
            weights.push_back(std::max(genomes_[member_idx].raw_fitness, 0.0F));
        }

        const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
        if (weight_sum == 0.0F)
        {
            std::uniform_int_distribution<size_t> distribution(0U, member_indices.size() - 1U);
            return member_indices[distribution(random_engine_)];
        }

        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        return member_indices[distribution(random_engine_)];
    }

    std::vector<Genome>         genomes_{};
    std::vector<FitnessTracker> fitness_trackers_{};
    std::vector<Species>        group_of_species_{};
    InnovationTracker           innovation_tracker_{};
    std::mt19937                random_engine_{};
    int32_t                     next_group_of_species_id_{0};
    uint32_t                    generation_index_{0U};
};
} // namespace neat
