#include "onnxruntime_cxx_api.h"
#include <opencv4/opencv2/opencv.hpp>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

const std::string IMGS_DIR = "/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images/test/";
constexpr int     HEIGHT   = 600;
constexpr int     WIDTH    = 600;

std::vector<std::string> getPngFilesInDirectory(const std::string &directory_path)
{
    std::vector<std::string> png_files;
    for (const auto &entry : std::filesystem::directory_iterator(directory_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
        {
            png_files.push_back(entry.path().string());
        }
    }
    return png_files;
}

int main()
{
    // Get all images in the directory
    auto imgs = getPngFilesInDirectory(IMGS_DIR);
    // Initialize an inference session with the ONNX model.
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "ModelInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    const std::string model_path = "semseg.onnx"; // update with actual model path
    Ort::Session      onnx_session(env, model_path.c_str(), session_options);

    for (const auto img_path : imgs)
    {
        cv::Mat img = cv::imread(img_path);
        assert(!img.empty() && "Could not open or find the image");

        // Resize the image to the size expected by the model.
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(HEIGHT, WIDTH));
        // cv::imshow("input", img);
        // cv::waitKey(0);

        // // Normalize the image pixels to be in the range [0, 1].
        // img.convertTo(img, CV_32F, 1.0 / 255);
        img.convertTo(img, CV_32FC3);
        img /= 255.0;

        // Create ONNX Runtime input tensor
        // Create ONNX Runtime input tensor
        size_t             input_tensor_size = 1 * 3 * HEIGHT * WIDTH;
        std::vector<float> input_tensor_values(input_tensor_size);
        float             *input_tensor_data = input_tensor_values.data();
        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < HEIGHT; ++h)
            {
                for (int w = 0; w < WIDTH; ++w)
                {
                    input_tensor_data[c * HEIGHT * WIDTH + h * WIDTH + w] = img.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // // Create input tensor object from data values.
        Ort::MemoryInfo        memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 4> input_shape{1, 3, img.rows, img.cols};
        Ort::Value             input_tensor = Ort::Value::CreateTensor<float>(
            memInfo, input_tensor_data, input_tensor_size, input_shape.data(), input_shape.size());

        // // Run the model on the input tensor and collect outputs.
        const char             *input_names[]  = {"input.1"};
        const char             *output_names[] = {"605"};
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

        seg *= 120; // normalize to range 0-255 if needed
        cv::imshow("seg", seg);
        cv::waitKey(10);
    }
    return 0;
}
