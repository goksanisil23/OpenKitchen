#include "DummyBrain.h"

#include <cstdio>

namespace dummy_brain
{

__global__ void initCurandStatesKernel(unsigned int seed, curandState *states)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize cuRAND state for each thread
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void decisionMakingKernel(Agent        *d_agents,
                                     Ray_         *d_rays,
                                     const int32_t num_agents,
                                     const int32_t agent_offset,
                                     const int32_t ray_offset,
                                     curandState  *curand_states)
{
    constexpr float kClosenessLimitSq = 100.f * 100.f;
    int             idx               = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_agents)
    {
        const int32_t agent_idx = agent_offset + idx;
        // Act based on the ray readings
        for (int i = 0; i < NUM_RAYS_PER_AGENT; ++i)
        {
            const int32_t ray_idx         = ray_offset + idx * NUM_RAYS_PER_AGENT + i;
            float         ray_dist        = d_rays[ray_idx].hit_dist;
            bool          is_hit_enemy    = (d_rays[ray_idx].hit_type == HitType::kEnemy);
            bool          is_hit_obstacle = (d_rays[ray_idx].hit_type == HitType::kObstacle);
            if (ray_dist < kClosenessLimitSq)
            {
                if (is_hit_enemy)
                {
                    // d_agents[agent_idx].heading += M_PI * curand_uniform(&curand_states[idx]);
                    d_agents[agent_idx].heading += curand_uniform(&curand_states[idx]);
                }
                else if (is_hit_obstacle)
                {
                    // d_agents[agent_idx].heading += (M_PI / 3.F) * curand_uniform(&curand_states[idx]);
                    d_agents[agent_idx].heading += curand_uniform(&curand_states[idx]);
                }
            }
        }
    }
}
void decisionMaking(const cudaStream_t &stream,
                    Agent              *d_agents,
                    Ray_               *d_rays,
                    const int32_t       threads_per_block,
                    const int32_t       blocks_per_grid,
                    const int32_t       num_agents,
                    const int32_t       agent_offset,
                    const int32_t       ray_offset,
                    curandState        *curand_states)
{
    // constexpr dim3 threadsPerBlock(16);
    // constexpr dim3 blocksPerGrid((NUM_AGENTS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    decisionMakingKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_agents, d_rays, num_agents, agent_offset, ray_offset, curand_states);
}

void initCurandStates(const int32_t threads_per_block, const int32_t blocks_per_grid, curandState *curand_states)
{
    initCurandStatesKernel<<<blocks_per_grid, threads_per_block>>>(time(NULL), curand_states);
}

} // namespace dummy_brain
