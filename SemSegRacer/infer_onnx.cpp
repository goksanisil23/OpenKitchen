#include "onnxruntime_cxx_api.h"
#include <opencv4/opencv2/opencv.hpp>

#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    // Initialize an inference session with the ONNX model.
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "ModelInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    const std::string model_path = "semseg.onnx"; // update with actual model path
    Ort::Session      onnx_session(env, model_path.c_str(), session_options);

    cv::Mat img = cv::imread("/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images/test/02299.png");
    assert(!img.empty() && "Could not open or find the image");

    // Resize the image to the size expected by the model.
    cv::resize(img, img, cv::Size(600, 600));
    cv::imshow("input", img);
    cv::waitKey(0);

    // // Normalize the image pixels to be in the range [0, 1].
    img.convertTo(img, CV_32F, 1.0 / 255);

    // // Prepare the input tensor.
    std::array<int64_t, 4> input_shape{1, 3, img.rows, img.cols};
    std::vector<float>     img_data;
    img_data.reserve(img.total() * img.channels());

    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            cv::Vec<float, 3> &pixel = img.at<cv::Vec<float, 3>>(y, x);
            for (int c = 0; c < img.channels(); ++c)
            {
                img_data.push_back(pixel[c]);
            }
        }
    }

    // // BGR to RGB.
    for (size_t i = 0; i < img_data.size(); i += 3)
    {
        std::swap(img_data[i], img_data[i + 2]);
    }

    // // Create input tensor object from data values.
    Ort::MemoryInfo memInfo      = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value      input_tensor = Ort::Value::CreateTensor<float>(
        memInfo, img_data.data(), img_data.size(), input_shape.data(), input_shape.size());

    // // Run the model on the input tensor and collect outputs.
    const char             *input_names[]  = {"input.1"};
    const char             *output_names[] = {"619"};
    std::vector<Ort::Value> output_tensors =
        onnx_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Get pointer to output tensor float data.
    float *floatarr = output_tensors[0].GetTensorMutableData<float>();

    // assuming output shape is {1, 3, height, width}
    // perform argmax along axis 1 (channels)
    cv::Mat seg = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            int   max_id  = 0;
            float max_val = floatarr[i * img.cols + j];
            for (int c = 1; c < 3; ++c)
            {
                float val = floatarr[c * img.rows * img.cols + i * img.cols + j];
                if (val > max_val)
                {
                    max_val = val;
                    max_id  = c;
                }
            }
            seg.at<uchar>(i, j) = max_id;
        }
    }

    seg *= 255; // normalize to range 0-255 if needed
    cv::imshow("seg", seg);
    cv::waitKey(0);

    return 0;
}
