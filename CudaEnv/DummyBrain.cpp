#include "DummyBrain.h"

namespace dummy_brain
{
DummyBrain::DummyBrain(const int32_t agent_offset, const int32_t num_agents)
    : agent_offset_{agent_offset}, num_agents_{num_agents}, ray_offset_{agent_offset_ * NUM_RAYS_PER_AGENT},
      blocks_per_grid_{(num_agents + kThreadsPerBlock - 1) / kThreadsPerBlock}
{
    cudaMalloc((void **)&curand_states_, sizeof(curandState) * kThreadsPerBlock * blocks_per_grid_);

    initCurandStates(kThreadsPerBlock, blocks_per_grid_, curand_states_);
}

void DummyBrain::step(const cudaStream_t &stream, Agent *d_agents, Ray_ *d_rays)
{
    // Decision making from last sensing + Move
    decisionMaking(stream,
                   d_agents,
                   d_rays,
                   kThreadsPerBlock,
                   blocks_per_grid_,
                   num_agents_,
                   agent_offset_,
                   ray_offset_,
                   curand_states_);
}
} // namespace dummy_brain