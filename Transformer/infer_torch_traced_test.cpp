#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

// Normalization constants matching Python
const float LASER_X_MIN = -200.0f;
const float LASER_X_MAX = 200.0f;
const float LASER_Y_MIN = -200.0f;
const float LASER_Y_MAX = 200.0f;

// Normalize a laser point (x, y) to [-1, 1]
std::vector<float> normalizeLaserPoint(float x, float y)
{
    float x_norm = 2 * (x - LASER_X_MIN) / (LASER_X_MAX - LASER_X_MIN) - 1;
    float y_norm = 2 * (y - LASER_Y_MIN) / (LASER_Y_MAX - LASER_Y_MIN) - 1;
    return {x_norm, y_norm};
}

// Denormalize controls (throttle in [0,100], steering in [-2,2])
std::vector<float> denormalizeControls(const std::vector<float> &norm)
{
    float throttle = (norm[0] + 1) / 2 * 100.0f;
    float steering = (norm[1] + 1) / 2 * 4.0f - 2.0f;
    return {throttle, steering};
}

int main()
{
    // Load the TorchScript model
    torch::jit::script::Module model =
        torch::jit::load("/home/s0001734/Downloads/OpenKitchen/Transformer/best_model_scripted.pt");
    model.to(torch::kCUDA);
    model.eval();

    // Example raw laser points (x, y); replace with your sensor data
    std::vector<std::pair<float, float>> rawLaserPoints = {{-150.0f, 50.0f},
                                                           {-100.0f, 75.0f},
                                                           {-50.0f, 100.0f},
                                                           {0.0f, 125.0f},
                                                           {50.0f, 100.0f},
                                                           {100.0f, 75.0f},
                                                           {150.0f, 50.0f}};

    // Normalize each laser point
    std::vector<float> normalizedPoints;
    for (const auto &pt : rawLaserPoints)
    {
        std::vector<float> norm = normalizeLaserPoint(pt.first, pt.second);
        normalizedPoints.insert(normalizedPoints.end(), norm.begin(), norm.end());
    }

    // Create an input tensor of shape [1, 7, 2]
    torch::Tensor input = torch::from_blob(normalizedPoints.data(), {1, 7, 2}, torch::kFloat).clone();
    input               = input.to(torch::kCUDA);

    // Run inference
    torch::Tensor output = model.forward({input}).toTensor();
    output               = output.to(torch::kCPU);

    // Extract and denormalize the predicted controls
    std::vector<float> normControls(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    std::vector<float> controls = denormalizeControls(normControls);
    std::cout << "Throttle: " << controls[0] << ", Steering: " << controls[1] << std::endl;

    return 0;
}
