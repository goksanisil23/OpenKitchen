#include "ScreenGrabber.h"

#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include <vector>

#include "fpng.h"

namespace
{
constexpr int kDownscaleFactor = 4;

void downsample(const uint8_t *src,
                const int      src_w,
                const int      src_h,
                std::vector<uint8_t> &dst,
                const int      factor)
{
    const int dst_w = src_w / factor;
    const int dst_h = src_h / factor;
    const int area  = factor * factor;
    dst.resize(static_cast<size_t>(dst_w) * dst_h * 4);

    for (int y = 0; y < dst_h; ++y)
    {
        for (int x = 0; x < dst_w; ++x)
        {
            uint32_t r = 0, g = 0, b = 0, a = 0;
            for (int dy = 0; dy < factor; ++dy)
            {
                for (int dx = 0; dx < factor; ++dx)
                {
                    const uint8_t *p = src + (static_cast<size_t>(y * factor + dy) * src_w + (x * factor + dx)) * 4;
                    r += p[0];
                    g += p[1];
                    b += p[2];
                    a += p[3];
                }
            }
            uint8_t *d = dst.data() + (static_cast<size_t>(y) * dst_w + x) * 4;
            d[0]       = static_cast<uint8_t>(r / area);
            d[1]       = static_cast<uint8_t>(g / area);
            d[2]       = static_cast<uint8_t>(b / area);
            d[3]       = static_cast<uint8_t>(a / area);
        }
    }
}

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

    void captureRenderedFrame()
    {
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, host_buffer_.data());
    }

    void saveRenderTargetToFile(const std::string &filename)
    {
        downsample(host_buffer_.data(), width_, height_, downsampled_buffer_, kDownscaleFactor);
        writePNG(filename.c_str(),
                 downsampled_buffer_.data(),
                 width_ / kDownscaleFactor,
                 height_ / kDownscaleFactor,
                 true);
    }

    std::vector<uint8_t> getRenderTargetHost()
    {
        return host_buffer_;
    }

    RenderTargetInfo getRenderTargetInfo() const
    {
        return render_target_info_;
    }

    void getRenderTargetDevice(void *dst, size_t dst_pitch)
    {
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
    const int width_;
    const int height_;

    std::vector<uint8_t> host_buffer_;
    std::vector<uint8_t> downsampled_buffer_;

    const ScreenGrabber::RenderTargetInfo render_target_info_;
};

ScreenGrabber::ScreenGrabber(const int width, const int height) : impl_(std::make_unique<Impl>(width, height))
{
}

ScreenGrabber::~ScreenGrabber() = default;

void ScreenGrabber::captureRenderedFrame()
{
    impl_->captureRenderedFrame();
}

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
