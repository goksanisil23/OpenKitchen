#include "ScreenGrabber.h"
#include <GL/glew.h>
#include <iostream>
#include <vector>

#include <GL/gl.h>
#include <chrono>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <fstream>

// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "fpng.h"

namespace
{

void writePNG(const char *path, const unsigned char *rgba, int w, int h, bool flip_y = true)
{
    static bool fpng_initialized = false;
    if (!fpng_initialized)
    {
        fpng_initialized = true;
        fpng::fpng_init();
    }
    const unsigned char       *src = rgba;
    std::vector<unsigned char> tmp;
    if (flip_y)
    {
        tmp.resize(size_t(w) * h * 4);
        for (int y = 0; y < h; ++y)
            memcpy(&tmp[size_t(y) * w * 4], &rgba[size_t(h - 1 - y) * w * 4], size_t(w) * 4);
        src = tmp.data();
    }
    if (!fpng::fpng_encode_image_to_file(path, src, w, h, 4))
        std::cerr << "FPNG write failed: " << path << "\n";
}

void saveCudaArrayToFileAsPPM(cudaArray_t cuda_array,
                              const int   w,
                              const int   h,
                              const char *path,
                              const bool  flip_y = true)
{
    std::vector<unsigned char> host_data(static_cast<size_t>(w * h * 4));
    const size_t               pitch{static_cast<size_t>(w) * 4}; // bytes per row on host
    if (cudaMemcpy2DFromArray(host_data.data(), pitch, cuda_array, 0, 0, pitch, h, cudaMemcpyDeviceToHost) ==
        cudaSuccess)
    {
        writePNG(path, host_data.data(), w, h, flip_y);
    }
    else
    {
        std::cerr << "Failed to copy data from CUDA array to host memory." << std::endl;
    }
}

} // namespace

class ScreenGrabber::Impl
{

  public:
    static constexpr int kChannels = 4;

    Impl(const unsigned int id, const int width, const int height)
        : width_(width), height_(height), render_target_info_{width, height, kChannels}
    {
        cudaGraphicsGLRegisterImage(&cuda_resource_, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    }

    ~Impl() = default;

    void saveRenderTargetToFile(const std::string &filename)
    {
        cudaGraphicsMapResources(1, &cuda_resource_, 0);
        cudaArray_t cuda_array;
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);

        saveCudaArrayToFileAsPPM(cuda_array, width_, height_, filename.c_str(), true);
    }

    std::vector<uint8_t> getRenderTargetHost()
    {
        std::vector<uint8_t> host_data(static_cast<size_t>(width_ * height_ * kChannels));
        const size_t         pitch{static_cast<size_t>(width_ * kChannels)}; // bytes per row on host

        cudaGraphicsMapResources(1, &cuda_resource_, 0);
        cudaArray_t cuda_array;
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);

        if (cudaMemcpy2DFromArray(host_data.data(), pitch, cuda_array, 0, 0, pitch, height_, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        {
            std::cerr << "Failed to copy data from CUDA array to host memory." << std::endl;
        }
        cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
        return host_data;
    }

    RenderTargetInfo getRenderTargetInfo() const
    {
        return render_target_info_;
    }

    void getRenderTargetDevice(void *dst, size_t dst_pitch)
    {
        cudaGraphicsMapResources(1, &cuda_resource_, 0);
        cudaArray_t cuda_array;
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);

        const size_t row_bytes = render_target_info_.row_bytes();
        const size_t height    = static_cast<size_t>(render_target_info_.height);

        cudaMemcpy2DFromArray(dst, dst_pitch, cuda_array, 0, 0, row_bytes, height, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
    }

  private:
    cudaGraphicsResource_t cuda_resource_;
    const int              width_;
    const int              height_;

    const ScreenGrabber::RenderTargetInfo render_target_info_;
};

ScreenGrabber::ScreenGrabber(const unsigned int id, const int width, const int height)
    : impl_(std::make_unique<Impl>(id, width, height))
{
}

ScreenGrabber::~ScreenGrabber() = default;

void ScreenGrabber::saveRenderTargetToFile(const std::string &filename)
{
    impl_->saveRenderTargetToFile(filename);
}

std::vector<uint8_t> ScreenGrabber::getRenderTargetHost() const
{
    return impl_->getRenderTargetHost();
}

ScreenGrabber::RenderTargetInfo ScreenGrabber::getRenderTargetInfo() const
{
    return impl_->getRenderTargetInfo();
}

void ScreenGrabber::getRenderTargetDevice(void *dst, size_t dst_pitch)
{
    impl_->getRenderTargetDevice(dst, dst_pitch);
}
