#include "raylib.h"

#include "Common.h"

#include <GL/glew.h>

#include <chrono>
#include <cmath>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

class Environment
{
  public:
    Environment();
    ~Environment();

    void step();

  private:
    void moveAgents();

  public:
    Agent *h_agents_;
    Agent *d_agents_;
    Ray_  *d_rays_;

    cudaStream_t stream_;

  private:
    cudaGraphicsResource_t cuda_resource_;
    RenderTexture2D        render_target_;
};