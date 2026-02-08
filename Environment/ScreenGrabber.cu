#include "ScreenGrabber.h"

#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include <vector>

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
} // namespace

class ScreenGrabber::Impl
{
  public:
    static constexpr int kChannels = 4;

    Impl(const int width, const int height) : width_(width), height_(height), render_target_info_{width, height, kChannels}
    {
        // Allocate host buffer for screen capture
        host_buffer_.resize(static_cast<size_t>(width) * height * kChannels);
    }

    ~Impl() = default;

    void saveRenderTargetToFile(const std::string &filename)
    {
        // Read pixels from OpenGL framebuffer (screen)
        captureScreen();
        // Flip Y because OpenGL has origin at bottom-left, but images have origin at top-left
        writePNG(filename.c_str(), host_buffer_.data(), width_, height_, true);
    }

    std::vector<uint8_t> getRenderTargetHost()
    {
        captureScreen();
        return host_buffer_;
    }

    RenderTargetInfo getRenderTargetInfo() const
    {
        return render_target_info_;
    }

    void getRenderTargetDevice(void *dst, size_t dst_pitch)
    {
        captureScreen();
        // Copy row by row from host buffer to device
        const size_t row_bytes = render_target_info_.row_bytes();
        for (int y = 0; y < height_; ++y)
        {
            cudaMemcpy(static_cast<uint8_t *>(dst) + y * dst_pitch,
                       host_buffer_.data() + y * row_bytes,
                       row_bytes,
                       cudaMemcpyHostToDevice);
        }
    }

  private:
    void captureScreen()
    {
        // Use OpenGL to read from the default framebuffer
        glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, host_buffer_.data());
    }

  private:
    const int width_;
    const int height_;

    std::vector<uint8_t> host_buffer_;

    const ScreenGrabber::RenderTargetInfo render_target_info_;
};

ScreenGrabber::ScreenGrabber(const int width, const int height) : impl_(std::make_unique<Impl>(width, height))
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
