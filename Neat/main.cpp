#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "Environment/Environment.h"
#include "NeatAgent.hpp"
#include "NeatConfig.hpp"
#include "NeatGenomeOps.hpp"
#include "NeatPopulation.hpp"
#include "NeatTypes.hpp"

constexpr bool kDrawRays{true};
constexpr bool kHiddenWindow{false};

constexpr int16_t kPopulationSize{50};

namespace
{
void resetAgents(Environment &env, const std::vector<std::unique_ptr<neat::NeatAgent>> &agents)
{
    for (const auto &agent : agents)
    {
        env.resetAgent(agent.get(), false, false, false);
    }
}

bool allAgentsDone(const std::vector<std::unique_ptr<neat::NeatAgent>> &agents)
{
    return std::all_of(
        agents.begin(), agents.end(), [](const std::unique_ptr<neat::NeatAgent> &agent) { return agent->isDone(); });
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    std::vector<std::unique_ptr<neat::NeatAgent>> agents{};
    agents.reserve(kPopulationSize);
    for (int16_t agent_idx = 0; agent_idx < kPopulationSize; ++agent_idx)
    {
        agents.push_back(std::make_unique<neat::NeatAgent>(Vec2d{0.0F, 0.0F}, 0.0F, agent_idx));
    }

    Environment env(argv[1], createBaseAgentPtrs(agents), kDrawRays, kHiddenWindow);

    neat::Population population{kPopulationSize};
    population.setAgents(agents);
    resetAgents(env, agents);

    uint32_t generation_idx{0U};
    while ((neat::Config::kMaxGenerations == 0U) || (generation_idx < neat::Config::kMaxGenerations))
    {
        population.resetFitnessTracking();
        env.step(); // Populate the initial raycast observations after reset.
        population.updateFitnessTracking(agents, *env.race_track_);

        uint32_t step_count{0U};
        while (!allAgentsDone(agents) && (step_count < neat::Config::kMaxStepsPerGeneration))
        {
            for (const auto &agent : agents)
            {
                if (!agent->isDone())
                {
                    agent->updateAction();
                }
            }

            env.step();
            population.updateFitnessTracking(agents, *env.race_track_);
            step_count++;
        }

        population.assignFitnessFromTracking();
        population.speciate();
        const neat::GenerationStats stats = population.getGenerationStats();

        std::cout << "generation=" << generation_idx << " best=" << stats.best_fitness
                  << " avg=" << stats.average_fitness << " species=" << stats.species_count << " steps=" << step_count
                  << std::endl;

        generation_idx++;

        population.evolve();
        population.setAgents(agents);
        resetAgents(env, agents);
    }

    return 0;
}
