#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "CollisionChecker.h"

__device__ bool raySegmentIntersect(const float ray_ox,
                                    const float ray_oy,
                                    const float ray_dx,
                                    const float ray_dy,
                                    const float seg_x1,
                                    const float seg_y1,
                                    const float seg_x2,
                                    const float seg_y2,
                                    const float ray_range,
                                    float      *out_t)
{
    const float seg_dx = seg_x2 - seg_x1;
    const float seg_dy = seg_y2 - seg_y1;
    const float denom  = ray_dx * seg_dy - ray_dy * seg_dx;

    if (fabsf(denom) < 1e-8F)
        return false; // Parallel

    const float t = ((seg_x1 - ray_ox) * seg_dy - (seg_y1 - ray_oy) * seg_dx) / denom;
    const float s = ((seg_x1 - ray_ox) * ray_dy - (seg_y1 - ray_oy) * ray_dx) / denom;

    if ((t >= 0.0F) && (t <= ray_range) && (s >= 0.0F) && (s <= 1.0F))
    {
        *out_t = t;
        return true;
    }
    return false;
}

__global__ void castRaysToSegmentsKernel(Ray_            *rays,
                                         const int        total_rays,
                                         const Segment2d *segments,
                                         const int        num_segments,
                                         const float      ray_range)
{
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= total_rays || !rays[ray_idx].active)
        return;

    const float ray_dx = cosf(rays[ray_idx].angle);
    const float ray_dy = sinf(rays[ray_idx].angle);
    float       min_t  = ray_range;

    for (int i = 0; i < num_segments; ++i)
    {
        float t;
        if (raySegmentIntersect(rays[ray_idx].x,
                                rays[ray_idx].y,
                                ray_dx,
                                ray_dy,
                                segments[i].x1,
                                segments[i].y1,
                                segments[i].x2,
                                segments[i].y2,
                                min_t,
                                &t))
        {
            min_t = t;
        }
    }

    rays[ray_idx].hit_x = rays[ray_idx].x + min_t * ray_dx;
    rays[ray_idx].hit_y = rays[ray_idx].y + min_t * ray_dy;
}

class CollisionChecker::Impl
{
  public:
    Impl(const Segment2d *d_segments, size_t num_segments, const std::vector<Agent *> &agents)
        : d_segments_(d_segments), num_segments_(num_segments)
    {
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

        cudaStreamDestroy(stream_);
    }

    void checkCollisionImpl()
    {
        runCollisionKernel(stream_);
        cudaStreamSynchronize(stream_);
    }

    const Ray_ *getHostRays() const
    {
        return h_rays_;
    }

    size_t getNumRays() const
    {
        return num_total_rays_;
    }

  private:
    void runCollisionKernel(cudaStream_t stream)
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

        castRaysToSegmentsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_rays_,
                                                                                static_cast<int>(num_total_rays_),
                                                                                d_segments_,
                                                                                static_cast<int>(num_segments_),
                                                                                Agent::kSensorRange);
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
                // Note: with vector intersection, we don't need to saturate to sensor range
                // as the kernel already handles this properly
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

    const Segment2d *d_segments_;
    size_t           num_segments_;

    cudaStream_t stream_;
};

CollisionChecker::CollisionChecker(const Segment2d *d_segments, size_t num_segments, const std::vector<Agent *> &agents)
    : impl_(std::make_unique<Impl>(d_segments, num_segments, agents))
{
}

CollisionChecker::~CollisionChecker() = default;

void CollisionChecker::checkCollision()
{
    impl_->checkCollisionImpl();
}

const Ray_ *CollisionChecker::getHostRays() const
{
    return impl_->getHostRays();
}

size_t CollisionChecker::getNumRays() const
{
    return impl_->getNumRays();
}
