#include <GL/glew.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "CollisionChecker.h"

__constant__ int kCudaLeftBarrierColor[4]  = {kLeftBarrierColor[0],
                                              kLeftBarrierColor[1],
                                              kLeftBarrierColor[2],
                                              kLeftBarrierColor[3]};
__constant__ int kCudaRightBarrierColor[4] = {kRightBarrierColor[0],
                                              kRightBarrierColor[1],
                                              kRightBarrierColor[2],
                                              kRightBarrierColor[3]};

__constant__ int kCudaScreenWidth  = kScreenWidth;
__constant__ int kCudaScreenHeight = kScreenHeight;

__device__ __forceinline__ bool isBarrierPixel(uchar4 data)
{
    return ((data.x == static_cast<unsigned char>(kCudaLeftBarrierColor[0])) &&
            (data.y == static_cast<unsigned char>(kCudaLeftBarrierColor[1])) &&
            (data.z == static_cast<unsigned char>(kCudaLeftBarrierColor[2])) &&
            (data.w == static_cast<unsigned char>(kCudaLeftBarrierColor[3]))) ||
           ((data.x == static_cast<unsigned char>(kCudaRightBarrierColor[0])) &&
            (data.y == static_cast<unsigned char>(kCudaRightBarrierColor[1])) &&
            (data.z == static_cast<unsigned char>(kCudaRightBarrierColor[2])) &&
            (data.w == static_cast<unsigned char>(kCudaRightBarrierColor[3])));
}

__device__ __forceinline__ bool checkCollisionAndDrawRays(cudaSurfaceObject_t surface,
                                                          Ray_               *rays,
                                                          const int           ray_idx,
                                                          const int           x,
                                                          const int           y,
                                                          const bool          draw_rays)
{
    uchar4 data;
    int    flipped_y = kCudaScreenHeight - y - 1; // since cuda surface is flipped w.r.t raylib coordinates in y
    surf2Dread(&data, surface, x * sizeof(uchar4), flipped_y);

    if (isBarrierPixel(data))
    {
        rays[ray_idx].hit_x = x;
        rays[ray_idx].hit_y = y;
        return true;
    }

    // Color the pixels along the ray, since they're not hit
    if (draw_rays)
    {
        data.x = 255;
        data.y = 255;
        data.z = 255;
        surf2Dwrite(data, surface, x * sizeof(uchar4), flipped_y);
    }
    return false;
}

__global__ void castRaysKernel(cudaSurfaceObject_t surface, Ray_ *rays, const int total_rays, const bool draw_rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_rays)
    {
        if (!rays[idx].active)
            return;

        int x0 = static_cast<int>(rays[idx].x);
        int y0 = static_cast<int>(rays[idx].y);
        int x1 = static_cast<int>(rays[idx].x + Agent::kSensorRange * cos(rays[idx].angle));
        int y1 = static_cast<int>(rays[idx].y + Agent::kSensorRange * sin(rays[idx].angle));

        auto dx = std::abs(x1 - x0);
        auto dy = std::abs(y1 - y0);

        int x = x0;
        int y = y0;

        int sx = (x0 > x1) ? -1 : 1;
        int sy = (y0 > y1) ? -1 : 1;

        // No hit
        rays[idx].hit_x = x1;
        rays[idx].hit_y = y1;

        if (dx > dy)
        {
            int err = dx / 2;
            while (x != x1)
            {
                if (x >= 0 && x < kCudaScreenWidth && y >= 0 && y < kCudaScreenHeight)
                // if (x >= 0 && y >= 0)
                {
                    if (checkCollisionAndDrawRays(surface, rays, idx, x, y, draw_rays))
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
                    // rays[idx].hit_dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
                    break;
                }
            }
        }
        else
        {
            int err = dy / 2;
            while (y != y1)
            {
                if (x >= 0 && x < kCudaScreenWidth && y >= 0 && y < kCudaScreenHeight)
                // if (x >= 0 && y >= 0)
                {
                    if (checkCollisionAndDrawRays(surface, rays, idx, x, y, draw_rays))
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
                    // rays[idx].hit_dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
                    break;
                }
            }
        }
    }
}

class CollisionChecker::Impl
{
  public:
    Impl(const unsigned int texture_id, const std::vector<Agent *> &agents, const bool draw_rays = true)
        : draw_rays_(draw_rays)
    {
        if (!glIsTexture(texture_id))
        {
            std::cerr << "Not a valid GL texture: " << texture_id << std::endl;
        }

        // Register OpenGL texture to CUDA
        cudaGraphicsGLRegisterImage(
            &cuda_resource_, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
        cudaStreamCreate(&stream_);

        agents_         = agents;
        num_total_rays_ = agents[0]->sensor_ray_angles_.size() * agents.size();

        cudaMallocHost(&h_rays_, num_total_rays_ * sizeof(Ray_));
        cudaMalloc(&d_rays_, num_total_rays_ * sizeof(Ray_));
    }

    ~Impl()
    {
        cudaFree(d_rays_);
        cudaFreeHost(h_rays_);

        cudaGraphicsUnregisterResource(cuda_resource_);
        cudaStreamDestroy(stream_);
    }

    void checkCollisionImpl()
    {
        // Map the resource (needs to occur whenever texture content changes)
        cudaGraphicsMapResources(1, &cuda_resource_, 0);
        cudaArray_t cuda_array;
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);

        cudaResourceDesc resource_desc = {};
        resource_desc.resType          = cudaResourceTypeArray;
        resource_desc.res.array.array  = cuda_array;
        cudaSurfaceObject_t surface;
        cudaCreateSurfaceObject(&surface, &resource_desc);

        runCollisionKernel(surface, stream_);

        cudaStreamSynchronize(stream_);
        cudaDestroySurfaceObject(surface);
        cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
    }

  private:
    void runCollisionKernel(cudaSurfaceObject_t surface, cudaStream_t stream)
    {
        for (size_t agent_idx = 0; agent_idx < agents_.size(); ++agent_idx)
        {
            const size_t num_rays = agents_[agent_idx]->sensor_ray_angles_.size();
            auto         agent    = agents_[agent_idx];
            for (size_t i = 0; i < num_rays; ++i)
            {
                h_rays_[agent_idx * num_rays + i].x =
                    agent->pos_.x + agent->sensor_offset_ * cos(kDeg2Rad * agent->rot_);
                h_rays_[agent_idx * num_rays + i].y =
                    agent->pos_.y + agent->sensor_offset_ * sin(kDeg2Rad * agent->rot_);
                h_rays_[agent_idx * num_rays + i].angle  = kDeg2Rad * (agent->rot_ + agent->sensor_ray_angles_[i]);
                h_rays_[agent_idx * num_rays + i].active = (!agent->crashed_);
            }
        }

        cudaMemcpy(d_rays_, h_rays_, num_total_rays_ * sizeof(Ray_), cudaMemcpyHostToDevice);

        constexpr dim3 threadsPerBlock(32);
        const dim3     blocksPerGrid((static_cast<int>(num_total_rays_) + threadsPerBlock.x - 1) / threadsPerBlock.x);

        castRaysKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            surface, d_rays_, static_cast<int>(num_total_rays_), draw_rays_);
        cudaGetLastError();

        cudaMemcpy(h_rays_, d_rays_, num_total_rays_ * sizeof(Ray_), cudaMemcpyDeviceToHost);

        for (size_t agent_idx = 0; agent_idx < agents_.size(); ++agent_idx)
        {
            const size_t num_rays      = agents_[agent_idx]->sensor_ray_angles_.size();
            auto         agent         = agents_[agent_idx];
            float        agent_rot_rad = agent->rot_ * kDeg2Rad;
            agent->sensor_hits_.clear();
            float min_dist2{Agent::kSensorRange * Agent::kSensorRange};
            for (size_t i = 0; i < num_rays; ++i)
            {
                // Calculate hit point relative to the robot
                Vec2d hit_pt_relative;
                float xTranslated = h_rays_[agent_idx * num_rays + i].hit_x - h_rays_[agent_idx * num_rays + i].x;
                float yTranslated = h_rays_[agent_idx * num_rays + i].hit_y - h_rays_[agent_idx * num_rays + i].y;
                hit_pt_relative.x = xTranslated * cos(agent_rot_rad) - yTranslated * sin(agent_rot_rad);
                hit_pt_relative.y = xTranslated * sin(agent_rot_rad) + yTranslated * cos(agent_rot_rad);
                if (hit_pt_relative.squaredNorm() > (Agent::kSensorRange * Agent::kSensorRange))
                {
                    // Saturate to sensor range if exceeds due to pixel rounding
                    hit_pt_relative.x = Agent::kSensorRange * cos(agents_[agent_idx]->sensor_ray_angles_[i]);
                    hit_pt_relative.y = Agent::kSensorRange * sin(agents_[agent_idx]->sensor_ray_angles_[i]);
                }
                agent->sensor_hits_.push_back(hit_pt_relative);
                if (hit_pt_relative.squaredNorm() < min_dist2)
                {
                    min_dist2 = hit_pt_relative.squaredNorm();
                }
            }
            constexpr float kCollisionThresholdDistSquared = 2.0f;
            if (min_dist2 < kCollisionThresholdDistSquared)
            {
                agent->crashed_ = true;
            }
        }
        cudaStreamSynchronize(stream);
    }

  private:
    std::vector<Agent *> agents_;

    Ray_ *h_rays_;
    Ray_ *d_rays_;

    size_t num_total_rays_;

    cudaGraphicsResource_t cuda_resource_;
    cudaStream_t           stream_;

    const bool draw_rays_{true};
};

CollisionChecker::CollisionChecker(const unsigned int          texture_id,
                                   const std::vector<Agent *> &agents,
                                   const bool                  draw_rays)
    : impl_(std::make_unique<Impl>(texture_id, agents, draw_rays))
{
}

CollisionChecker::~CollisionChecker() = default;

void CollisionChecker::checkCollision()
{
    impl_->checkCollisionImpl();
}
