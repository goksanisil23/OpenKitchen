#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Common.h"

namespace dummy_brain
{
class DummyBrain
{
  public:
    static constexpr int32_t kThreadsPerBlock = 16;

    DummyBrain(const int32_t agent_offset, const int32_t num_agents);

    void step(const cudaStream_t &stream, Agent *d_agents, Ray_ *d_rays);

  private:
    const int32_t agent_offset_{};
    const int32_t num_agents_{};
    const int32_t ray_offset_{};

    curandState *curand_states_ = NULL;

    const int32_t blocks_per_grid_{};
};

void decisionMaking(const cudaStream_t &stream,
                    Agent              *d_agents,
                    Ray_               *d_rays,
                    const int32_t       threads_per_block,
                    const int32_t       blocks_per_grid,
                    const int32_t       num_agents,
                    const int32_t       agent_offset,
                    const int32_t       ray_offset,
                    curandState        *curant_states);

void initCurandStates(const int32_t threads_per_block, const int32_t blocks_per_grid, curandState *curand_states);

} // namespace dummy_brain