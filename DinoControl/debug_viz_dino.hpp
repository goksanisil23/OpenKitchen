#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

void showDinoPatchHeatmap(torch::jit::script::Module &dino_model,
                          const torch::Tensor        &dino_input_norm, // 1x3x224x224 (normalized), on CUDA or CPU
                          const std::string          &win_name = "dino_patches")
{
    using namespace torch::indexing;
    torch::NoGradGuard ng;

    torch::Tensor out = dino_model.run_method("forward_features", dino_input_norm).toTensor();

    // Need tokens: [1, N, C] (typically [1,197,384])
    if (out.dim() != 3 || out.size(0) != 1)
    {
        std::cout << "Not token output (expected [1,N,C]). Export a wrapper that returns forward_features().\n";
        return;
    }

    // Drop CLS token -> [196, C]
    auto      patch_tokens = out.index({0, Slice(1, None), Slice()}).contiguous(); // (N-1)xC
    const int num_patches  = (int)patch_tokens.size(0);
    const int grid         = (int)std::lround(std::sqrt((double)num_patches));

    if (grid * grid != num_patches)
    {
        std::cout << "Unexpected patch count: " << num_patches << " (not a square).\n";
        return;
    }

    // [grid, grid, C]
    auto patch_grid = patch_tokens.view({grid, grid, patch_tokens.size(1)});

    // Scalar heatmap per patch: L2 norm -> [grid, grid]
    auto heat = patch_grid.pow(2).sum(-1).sqrt();

    // Normalize to [0,255] uint8
    heat         = heat - heat.min();
    heat         = heat / (heat.max() + 1e-6);
    auto heat_u8 = (heat * 255.0).to(torch::kUInt8).cpu().contiguous();

    // OpenCV display
    cv::Mat heat_small(grid, grid, CV_8UC1, heat_u8.data_ptr<uint8_t>());
    cv::Mat heat_big;
    cv::resize(heat_small, heat_big, cv::Size(224, 224), 0, 0, cv::INTER_NEAREST);

    cv::Mat heat_color;
    cv::applyColorMap(heat_big, heat_color, cv::COLORMAP_JET);

    cv::imshow(win_name, heat_color);
    cv::waitKey(1);
}
