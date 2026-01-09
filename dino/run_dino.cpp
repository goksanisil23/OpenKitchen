#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

// Normalize with mean/std = 0.5 (same as minimal.py).
torch::Tensor preprocess(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224));

    // Convert to float32 in range [0,1].
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(float_img.data, {1, float_img.rows, float_img.cols, 3}, torch::kFloat32);

    // HWC -> CHW and normalize.
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = (tensor - 0.5) / 0.5;
    return tensor.clone(); // clone to own the memory.
}

int main(int argc, char *argv[])
{

    const std::string model_path = "../dino_vits16_ts.pt";
    const std::string image_path =
        "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/test_dir/birdseye_ZANDVOORT_0_0.png";

    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();
    std::cout << "Loaded model from " << model_path << "\n";

    // Run inference on own image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Failed to read image: " << image_path << "\n";
        return 1;
    }

    torch::Tensor      input = preprocess(img);
    torch::NoGradGuard no_grad;
    auto               output = module.forward({input}).toTensor();

    std::cout << "Output shape: " << output.sizes() << "\n";
    std::cout << "Sample output (first 8 vals): " << output.flatten().slice(0, 0, 8) << "\n";

    return 0;
}
