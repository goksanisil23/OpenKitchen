#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class ScreenGrabber
{
  public:
    ScreenGrabber(const unsigned int id, const int width, const int height);
    ~ScreenGrabber();

    void saveRenderTargetToFile(const std::string &filename);

    std::vector<uint8_t> getRenderTarget() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
