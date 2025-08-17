#pragma once
#include <memory>
#include <string>

class ScreenGrabber
{
  public:
    ScreenGrabber(const unsigned int id, const int width, const int height);
    ~ScreenGrabber();

    void saveRenderTargetToFile(const std::string &filename);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
