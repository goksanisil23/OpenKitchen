#include "Environment.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <GL/glew.h>
#include <chrono>
#include <cmath>
#include <string>

#include "Common.h"
#include "raylib.h"

namespace
{
constexpr Color OBSTACLE_COLOR = (Color){230, 41, 55, 255}; // red
constexpr Color RAY_COLOR      = (Color){253, 249, 0, 255}; // yellow
constexpr Color TEAM_A_COLOR   = (Color){0, 121, 241, 255}; // blue
constexpr Color TEAM_B_COLOR   = (Color){0, 228, 48, 255};  // green
constexpr float kBorderThickness{2.0};

constexpr float kInitVelocity = 0.09;
constexpr float kRayRange     = 100.F;
constexpr float kRayRangeSq   = kRayRange * kRayRange;
constexpr float kAgentFov     = 60.F;

constexpr float kAgentRadius = 10;
// applied against rounding error in bresenham, when starting pixel of the ray falls inside the robot
constexpr float kRayOffsetEps = kRayRange / 100.F;

} // namespace

__global__ void moveAgentsKernel(Agent *agents, Ray_ *rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Update robot position and angle
    agents[idx].x += agents[idx].velocity * cos(agents[idx].heading);
    agents[idx].y += agents[idx].velocity * sin(agents[idx].heading);

    // Move the rays
    for (int i = 0; i < NUM_RAYS_PER_AGENT; ++i)
    {
        float angle =
            agents[idx].heading - (kAgentFov / 2.F * DEG2RAD) + i * (kAgentFov * DEG2RAD / (NUM_RAYS_PER_AGENT - 1));
        // Ray information is reset each tick here as well.
        rays[idx * NUM_RAYS_PER_AGENT + i] = {agents[idx].x + (kAgentRadius + kRayOffsetEps) * cos(agents[idx].heading),
                                              agents[idx].y + (kAgentRadius + kRayOffsetEps) * sin(agents[idx].heading),
                                              angle,
                                              -1.F,
                                              HitType::kUndefined};
    }
}

__device__ __forceinline__ bool checkCollisionAndDrawRays(cudaSurfaceObject_t surface,
                                                          Ray_               *rays,
                                                          const int           idx,
                                                          const int           x,
                                                          const int           y,
                                                          const int           x0,
                                                          const int           y0,
                                                          const Color         enemy_color)
{
    uchar4 data;
    surf2Dread(&data, surface, x * sizeof(uchar4), y);

    if ((data.x == enemy_color.r && data.y == enemy_color.g && data.z == enemy_color.b))
    {
        // Squared hit distance
        rays[idx].hit_dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
        rays[idx].hit_type = HitType::kEnemy;
        return true;
    }
    else if (((data.x == OBSTACLE_COLOR.r && data.y == OBSTACLE_COLOR.g && data.z == OBSTACLE_COLOR.b)))
    {
        // Squared hit distance
        rays[idx].hit_dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
        rays[idx].hit_type = HitType::kObstacle;
        return true;
    }
    // Color the pixels along the ray
    data.x = RAY_COLOR.r;
    data.y = RAY_COLOR.g;
    data.z = RAY_COLOR.b;
    surf2Dwrite(data, surface, x * sizeof(uchar4), y);
    return false;
}

__global__ void castRays(cudaSurfaceObject_t surface, Ray_ *rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_TOTAL_RAYS)
    {
        const Color enemy_color = (idx < (NUM_RAYS_PER_AGENT * NUM_AGENTS_TEAM_A)) ? TEAM_B_COLOR : TEAM_A_COLOR;

        int x0 = static_cast<int>(rays[idx].x);
        int y0 = static_cast<int>(rays[idx].y);
        int x1 = static_cast<int>(rays[idx].x + kRayRange * cos(rays[idx].angle));
        int y1 = static_cast<int>(rays[idx].y + kRayRange * sin(rays[idx].angle));

        auto dx = std::abs(x1 - x0);
        auto dy = std::abs(y1 - y0);

        int x = x0;
        int y = y0;

        int sx = (x0 > x1) ? -1 : 1;
        int sy = (y0 > y1) ? -1 : 1;

        // No hit
        rays[idx].hit_dist = kRayRange * kRayRange;

        if (dx > dy)
        {
            int err = dx / 2;
            while (x != x1)
            {
                if (x >= 0 && x < SCREEN_WIDTH && y >= 0 && y < SCREEN_HEIGHT)
                {
                    if (checkCollisionAndDrawRays(surface, rays, idx, x, y, x0, y0, enemy_color))
                        break;

                    err -= dy;
                    if (err < 0)
                    {
                        y += sy;
                        err += dx;
                    }
                    x += sx;
                }
                else
                {
                    rays[idx].hit_dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
                    break;
                }
            }
        }
        else
        {
            int err = dy / 2;
            while (y != y1)
            {
                if (x >= 0 && x < SCREEN_WIDTH && y >= 0 && y < SCREEN_HEIGHT)
                {
                    if (checkCollisionAndDrawRays(surface, rays, idx, x, y, x0, y0, enemy_color))
                        break;

                    err -= dx;
                    if (err < 0)
                    {
                        x += sx;
                        err += dy;
                    }
                    y += sy;
                }
                else
                {
                    rays[idx].hit_dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
                    break;
                }
            }
        }
    }
}

Environment::Environment()
{
    // Agent resource initialization
    cudaMallocHost(&h_agents_, NUM_AGENTS * sizeof(Agent));
    const int min_start_y{50};
    const int max_start_y{SCREEN_HEIGHT - 50};
    // 1st team
    for (int32_t i = 0; i < NUM_AGENTS_TEAM_A; ++i)
    {
        h_agents_[i].x        = 50.F;
        h_agents_[i].y        = min_start_y + rand() % (max_start_y - min_start_y + 1);
        h_agents_[i].heading  = 0.F * DEG2RAD;
        h_agents_[i].velocity = kInitVelocity;
        printf("a: %f\n", h_agents_[i].y);
    }
    // 2nd team
    for (int32_t i = 0; i < NUM_AGENTS_TEAM_B; ++i)
    {
        h_agents_[i + NUM_AGENTS_TEAM_A].x        = SCREEN_WIDTH - 50.F;
        h_agents_[i + NUM_AGENTS_TEAM_A].y        = min_start_y + rand() % (max_start_y - min_start_y + 1);
        h_agents_[i + NUM_AGENTS_TEAM_A].heading  = 180.F * DEG2RAD;
        h_agents_[i + NUM_AGENTS_TEAM_A].velocity = kInitVelocity;
        printf("b: %f\n", h_agents_[i].y);
    }

    cudaMalloc(&d_agents_, NUM_AGENTS * sizeof(Agent));
    cudaMemcpy(d_agents_, h_agents_, NUM_AGENTS * sizeof(Agent), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rays_, NUM_TOTAL_RAYS * sizeof(Ray_));

    // Visualization initialization
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Raylib CUDA");
    // Create a render texture with Raylib
    render_target_ = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);
    // Register OpenGL texture to CUDA
    cudaGraphicsGLRegisterImage(
        &cuda_resource_, render_target_.texture.id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

    cudaStreamCreate(&stream_);
}

Environment::~Environment()
{
    // Unregister and clean up CUDA resources
    cudaGraphicsUnregisterResource(cuda_resource_);
    UnloadRenderTexture(render_target_);
    CloseWindow();
    cudaStreamDestroy(stream_);
}

void Environment::moveAgents()
{
    constexpr dim3 threadsPerBlock(16);
    constexpr dim3 blocksPerGrid((NUM_AGENTS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    moveAgentsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(d_agents_, d_rays_);
}

void Environment::step()
{
    moveAgents();

    cudaStreamSynchronize(stream_);
    cudaMemcpy(h_agents_, d_agents_, NUM_AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);

    // Draw into the texture with Raylib
    BeginTextureMode(render_target_);
    ClearBackground(BLACK);
    for (int i{0}; i < NUM_AGENTS; i++)
    {
        const Color agent_color = (i < NUM_AGENTS_TEAM_A) ? TEAM_A_COLOR : TEAM_B_COLOR;
        DrawCircleV({h_agents_[i].x, SCREEN_HEIGHT - h_agents_[i].y}, kAgentRadius, agent_color);
        DrawText(std::to_string(i).c_str(), h_agents_[i].x, SCREEN_HEIGHT - h_agents_[i].y, 10, WHITE);
    }

    // Draw map boundary
    DrawRectangleLinesEx((Rectangle){0, 0, SCREEN_WIDTH, SCREEN_HEIGHT}, kBorderThickness, RED);
    // Draw red boxes as obstacles
    DrawRectangle(300, 200, 100, 100, OBSTACLE_COLOR);
    DrawRectangle(500, 300, 100, 100, OBSTACLE_COLOR);
    EndTextureMode();

    // Map the resource (needs to occur whenever texture content changes)
    cudaGraphicsMapResources(1, &cuda_resource_, 0);
    cudaArray_t cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);

    // Create CUDA surface object
    cudaResourceDesc resource_desc = {};
    resource_desc.resType          = cudaResourceTypeArray;
    resource_desc.res.array.array  = cuda_array;
    cudaSurfaceObject_t surface;
    cudaCreateSurfaceObject(&surface, &resource_desc);

    // Launch CUDA kernel
    constexpr dim3 threadsPerBlock(16);
    constexpr dim3 blocksPerGrid((NUM_TOTAL_RAYS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    castRays<<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(surface, d_rays_);
    cudaStreamSynchronize(stream_);

    // Destroy CUDA surface object
    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cuda_resource_, 0);

    // Draw the texture to the screen
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawTextureRec(
        render_target_.texture,
        (Rectangle){
            0, 0, static_cast<float>(render_target_.texture.width), static_cast<float>(-render_target_.texture.height)},
        (Vector2){0, 0},
        WHITE);
    DrawFPS(10, 10);
    EndDrawing();
}
