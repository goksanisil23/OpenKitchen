#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class ScreenGrabber
{
  public:
    struct RenderTargetInfo
    {
        int width;
        int height;
        int channels = 4; // RGBA8

        size_t row_bytes() const
        {
            return static_cast<size_t>(width) * channels;
        }
    };

    ScreenGrabber(const int width, const int height);
    ~ScreenGrabber();

    void saveRenderTargetToFile(const std::string &filename);

    std::vector<uint8_t> getRenderTargetHost() const;
    void                 getRenderTargetDevice(void *dst, size_t dst_pitch);

    RenderTargetInfo getRenderTargetInfo() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
