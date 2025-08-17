#include "ScreenGrabber.h"
#include <GL/glew.h>
#include <iostream>
#include <vector>

#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

namespace
{

bool writePPM_P6(const char *path, const unsigned char *src, int w, int h, bool flip_y = true, bool is_bgra = false)
{
    FILE *f = std::fopen(path, "wb");
    if (!f)
        return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);

    std::vector<unsigned char> row((size_t)w * 3);
    for (int y = 0; y < h; ++y)
    {
        int                  sy = flip_y ? (h - 1 - y) : y;
        const unsigned char *s  = src + (size_t)sy * w * 4;
        unsigned char       *d  = row.data();

        for (int x = 0; x < w; ++x)
        { // RGBA -> RGB
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            s += 4;
            d += 3;
        }

        if (std::fwrite(row.data(), 1, row.size(), f) != row.size())
        {
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return true;
}

void saveCudaArrayToFileAsPPM(cudaArray_t cuda_array,
                              const int   w,
                              const int   h,
                              const char *path,
                              const bool  flip_y      = true,
                              const bool  src_is_bgra = false)
{
    std::vector<unsigned char> host_data(static_cast<size_t>(w * h * 4));
    const size_t               pitch{static_cast<size_t>(w) * 4}; // bytes per row on host
    if (cudaMemcpy2DFromArray(host_data.data(), pitch, cuda_array, 0, 0, pitch, h, cudaMemcpyDeviceToHost) ==
        cudaSuccess)
    {
        writePPM_P6(path, host_data.data(), w, h, flip_y, src_is_bgra);
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
    Impl(const unsigned int id, const int width, const int height) : width_(width), height_(height)
    {
        cudaGraphicsGLRegisterImage(&cuda_resource_, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    }

    ~Impl() = default;

    void saveRenderTargetToFile(const std::string &filename)
    {
        cudaGraphicsMapResources(1, &cuda_resource_, 0);
        cudaArray_t cuda_array;
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);

        saveCudaArrayToFileAsPPM(cuda_array, width_, height_, filename.c_str(), true, false);
    }

  private:
    cudaGraphicsResource_t cuda_resource_;
    const int              width_;
    const int              height_;
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