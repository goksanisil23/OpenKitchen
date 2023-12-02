#include "GeneticAgent.hpp"
#include "Network.hpp"

namespace genetic
{
// Mates 2 agents s.t. the offspring node is the weighted average of the parents' nodes
Network mate2AgentsAvg(const GeneticAgent &agent_1, const GeneticAgent &agent_2)
{
    constexpr float kMutationProb        = 0.05; // 0.05 probability for a given weight to mutate randomly
    constexpr float kDominantAgentWeight = 0.75; // 0.75 favoribility of the dominant agent during mating
    constexpr float kPassiveAgentWeight  = 1.F - kDominantAgentWeight;
    Network         offspring_nn;

    // superior agent
    const Network &n1 = (agent_1.score_ > agent_2.score_) ? agent_1.nn_ : agent_2.nn_;
    // inferior agent
    const Network &n2 = (agent_1.score_ > agent_2.score_) ? agent_2.nn_ : agent_1.nn_;

    std::mt19937                     rand_gen(std::random_device{}());
    std::uniform_real_distribution<> rand_dist(0.0, 1.0);

    // For each weight coefficient in the network, we'll either pick random mutation, or one of the coefficient from the parents
    for (int32_t i{0}; i < n1.weights_1_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_1_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else
        {
            offspring_nn.weights_1_.data()[i] =
                n1.weights_1_.data()[i] * kDominantAgentWeight + n2.weights_1_.data()[i] * kPassiveAgentWeight;
        }
    }
    for (int32_t i{0}; i < n1.weights_2_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_2_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else
        {
            offspring_nn.weights_2_.data()[i] =
                n1.weights_2_.data()[i] * kDominantAgentWeight + n2.weights_2_.data()[i] * kPassiveAgentWeight;
        }
    }

    return offspring_nn;
}

// Mate 2 agents s.t. the offsprint node is direct transfer of one of the parent's node, whose probability depends on the episode score
Network mate2AgentsSelective(const GeneticAgent &agent_1, const GeneticAgent &agent_2)
{
    constexpr float kMutationProb        = 0.1;  // 0.05 probability for a given weight to mutate randomly
    constexpr float kDominantAgentThresh = 0.75; // 0.75 favoribility of the dominant agent during mating
    Network         offspring_nn;

    // superior agent
    const Network &n1 = (agent_1.score_ > agent_2.score_) ? agent_1.nn_ : agent_2.nn_;
    // inferior agent
    const Network &n2 = (agent_1.score_ > agent_2.score_) ? agent_2.nn_ : agent_1.nn_;

    std::mt19937                     rand_gen(std::random_device{}());
    std::uniform_real_distribution<> rand_dist(0.0, 1.0);

    // For each weight coefficient in the network, we'll either pick random mutation, or one of the coefficient from the parents
    for (int32_t i{0}; i < n1.weights_1_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_1_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else if (rand_dist(rand_gen) < kDominantAgentThresh) // pick 1st parent
        {
            offspring_nn.weights_1_.data()[i] = n1.weights_1_.data()[i];
        }
        else
        {
            offspring_nn.weights_1_.data()[i] = n2.weights_1_.data()[i];
        }
    }
    for (int32_t i{0}; i < n1.weights_2_.size(); i++)
    {
        if (rand_dist(rand_gen) < kMutationProb)
        {
            offspring_nn.weights_2_.data()[i] = (rand_dist(rand_gen) - 0.5F) * 2.F; // scale from [0,1] to [-1,1]
        }
        else if (rand_dist(rand_gen) < kDominantAgentThresh) // pick 1st parent
        {
            offspring_nn.weights_2_.data()[i] = n1.weights_2_.data()[i];
        }
        else
        {
            offspring_nn.weights_2_.data()[i] = n2.weights_2_.data()[i];
        }
    }

    return offspring_nn;
}

/* *** Mating strategy ***
- 1 clone of the fittest agent always goes into next generation
- 1 self-mutation of the fittest agent always goes into next generation
- Total size of the colony is kept constant
- Only top N-agents are chosen for mating
- Probability to be chosen for mating is proportional to the agent score
*/
void chooseAndMateAgents(std::vector<GeneticAgent> &agents)
{
    const size_t                         colony_size{agents.size()}; // original size of the colony
    std::vector<float>                   agent_scores;
    std::unordered_map<int32_t, int32_t> agent_to_num_chosen_map; // which agent is chosen how many times

    // Filter so that only the top N scores can be parents
    std::sort(agents.begin(),
              agents.end(),
              [](const auto &agent_1, const auto &agent_2) { return agent_1.score_ > agent_2.score_; });
    constexpr size_t kNumParents{5};
    agents.resize(kNumParents);

    for (int32_t i{0}; i < static_cast<int32_t>(kNumParents); i++)
    {
        agent_scores.push_back(agents[i].score_);
        agent_to_num_chosen_map.insert({i, 0});
    }

    // Top agent clone and mutation are always chosen at least once
    std::vector<Network> new_networks;
    new_networks.push_back(agents.front().nn_);
    new_networks.emplace_back(mate2AgentsSelective(agents.front(), agents.front()));
    agent_to_num_chosen_map.at(0) += 2;
    std::cout << "top: " << agents.front().score_ << std::endl;

    // Probability of each agent to into mating is proportional to it's score
    std::random_device           rand_device;
    std::mt19937                 rand_generator(rand_device());
    std::discrete_distribution<> dist(agent_scores.begin(), agent_scores.end());

    while (new_networks.size() < colony_size)
    {
        int32_t first_agent = dist(rand_generator);
        agent_to_num_chosen_map.at(first_agent)++;
        // Prevent self-mutation
        int32_t second_agent{-1};
        while ((second_agent == -1) || (first_agent == second_agent))
        {
            second_agent = dist(rand_generator);
        }
        agent_to_num_chosen_map.at(second_agent)++;
        // Mate the chosen agents
        new_networks.emplace_back(mate2AgentsSelective(agents.at(first_agent), agents.at(second_agent)));
    }
    // Summarize who's chosen how many times
    for (size_t id{0}; id < agents.size(); id++)
    {
        std::cout << std::left << std::setw(15) << "agent score: " << std::right << std::setw(6) << agents[id].score_
                  << " # chosen: " << std::right << std::setw(6) << agent_to_num_chosen_map.at(id) << std::endl;
    }

    agents.resize(colony_size);
    for (size_t id{0}; id < agents.size(); id++)
    {
        agents[id].id_ = static_cast<int16_t>(id);
        agents[id].nn_ = std::move(new_networks[id]);
    }
}
} // namespace genetic