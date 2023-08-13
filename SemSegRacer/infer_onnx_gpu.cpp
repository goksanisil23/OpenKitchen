#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

constexpr bool    kuseCUDA{true};
const std::string IMGS_DIR         = "/home/raw_images/";
constexpr int     HEIGHT           = 600;
constexpr int     WIDTH            = 600;
constexpr int     BATCH_SIZE       = 1;
constexpr int     IN_CHANNEL_SIZE  = 3; // R,B,G
constexpr int     OUT_CHANNEL_SIZE = 4; // 0=background, 1=left, 2=right lane boundary, 3= drivable area

// Reads the input images from a directory
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

// Converts the image to the format expected by the network
void convertAndNormalizeImg(cv::Mat &img)
{
    // Resize the image to the size expected by the model.
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(HEIGHT, WIDTH));

    // Normalize the image pixels to be in the range [0, 1].
    img.convertTo(img, CV_32FC3);
    img /= 255.0;
}

// Creates ONNX Runtime input tensor
void generateFlatTensorData(const cv::Mat &img, std::vector<float> &flat_input_vec)
{
    // This model expects NCHW format (Batch, Channels, Height, Width) for input serialization
    size_t input_tensor_size = BATCH_SIZE * IN_CHANNEL_SIZE * HEIGHT * WIDTH;
    flat_input_vec.resize(input_tensor_size);
    float *input_tensor_data = flat_input_vec.data();
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
}

// Given the tensor output pointer, uses argmax to get the highest class and generate segmented image
void convertOutputAndShow(const float *output_array)
{
    // assuming output shape is {1, 3, height, width}
    // perform argmax along axis 1 (channels)
    cv::Mat seg = cv::Mat::zeros(HEIGHT, WIDTH, CV_8U);
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            int   max_id  = 0;
            float max_val = output_array[i * WIDTH + j];
            for (int c = 1; c < OUT_CHANNEL_SIZE; ++c)
            {
                float val = output_array[c * HEIGHT * WIDTH + i * WIDTH + j];
                if (val > max_val)
                {
                    max_val = val;
                    max_id  = c;
                }
            }

            // Assign color class based on highest probably class
            if ((max_id == 1) || (max_id == 2))
                seg.at<uchar>(i, j) = 250;
            else if (max_id == 3)
                seg.at<uchar>(i, j) = 120;
        }
    }

    cv::imshow("seg", seg);
    cv::waitKey(10);
}

int main()
{
    // Get all images in the directory
    auto imgs = getPngFilesInDirectory(IMGS_DIR);
    // Initialize an inference session with the ONNX model.
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "ModelInference");
    Ort::SessionOptions session_options;
    if (kuseCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    const std::string model_path = "semseg.onnx"; // update with actual model path
    Ort::Session      onnx_session(env, model_path.c_str(), session_options);

    for (const auto img_path : imgs)
    {
        cv::Mat img = cv::imread(img_path);
        assert(!img.empty() && "Could not open or find the image");

        convertAndNormalizeImg(img);

        std::vector<float> flat_input_vec;
        generateFlatTensorData(img, flat_input_vec);

        // Create input tensor object from flattened data
        Ort::MemoryInfo memInfo =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::array<int64_t, 4> input_shape{BATCH_SIZE, IN_CHANNEL_SIZE, HEIGHT, WIDTH};
        Ort::Value             input_tensor = Ort::Value::CreateTensor<float>(
            memInfo, flat_input_vec.data(), flat_input_vec.size(), input_shape.data(), input_shape.size());

        // Run the model on the input tensor and collect outputs.
        const char             *input_names[]  = {onnx_session.GetInputName(0, Ort::AllocatorWithDefaultOptions())};
        const char             *output_names[] = {onnx_session.GetOutputName(0, Ort::AllocatorWithDefaultOptions())};
        auto                    t0             = std::chrono::steady_clock::now();
        std::vector<Ort::Value> output_tensors =
            onnx_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto t1        = std::chrono::steady_clock::now();
        auto infer_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "inference: " << infer_dur << " ms." << std::endl;

        // Get pointer to output tensor float data.
        float *output_array = output_tensors[0].GetTensorMutableData<float>();

        convertOutputAndShow(output_array);
    }

    return 0;
}
