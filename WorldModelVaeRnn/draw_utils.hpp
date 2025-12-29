#include <algorithm>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

cv::Mat barPlotFromTensor(const torch::Tensor &tensor, int imgWidth = 640, int imgHeight = 480)
{
    // Move to CPU and flatten to 1D
    torch::Tensor t = tensor.detach().to(torch::kCPU).view(-1);
    int           n = t.size(0);

    // Get raw pointer
    const float *data = t.data_ptr<float>();

    // Compute min / max for scaling
    float minVal = data[0], maxVal = data[0];
    for (int i = 1; i < n; ++i)
    {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
    }
    if (maxVal == minVal)
    {
        maxVal = minVal + 1e-6f; // avoid division by zero
    }

    // Create white image
    cv::Mat img(imgHeight, imgWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    int barWidth = std::max(1, imgWidth / n);
    int bottom   = imgHeight - 10; // some margin

    for (int i = 0; i < n; ++i)
    {
        float v         = data[i];
        float norm      = (v - minVal) / (maxVal - minVal); // 0..1
        int   barHeight = static_cast<int>(norm * (imgHeight - 20));

        int x1 = i * barWidth;
        int y1 = bottom;
        int x2 = x1 + barWidth - 1;
        int y2 = bottom - barHeight;

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 0), cv::FILLED);
    }

    // Draw min/max as text
    cv::putText(img,
                "min: " + std::to_string(minVal),
                cv::Point(10, imgHeight - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 255),
                1);

    cv::putText(img,
                "max: " + std::to_string(maxVal),
                cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 255),
                1);

    return img;
}