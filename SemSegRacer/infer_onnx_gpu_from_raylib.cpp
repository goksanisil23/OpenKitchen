#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include "raylib_img_msg.h"
#include "spmc_queue.h"

constexpr bool kuseCUDA{true};
constexpr int  INFER_HEIGHT     = 600;
constexpr int  INFER_WIDTH      = 600;
constexpr int  IN_IMG_WIDTH     = 1000;
constexpr int  IN_IMG_HEIGHT    = 1000;
constexpr int  BATCH_SIZE       = 1;
constexpr int  IN_CHANNEL_SIZE  = 3; // R,B,G
constexpr int  OUT_CHANNEL_SIZE = 4; // 0=background, 1=left, 2=right lane boundary, 3= drivable area

// Converts the serialized raylib image to opencv for visualization
cv::Mat convertRaylibToOpencv(const std::array<uint8_t, IN_IMG_WIDTH * IN_IMG_HEIGHT * 4> &raylib_image)
{
    constexpr int kRowWidth = IN_IMG_WIDTH * 4;
    // assuming output shape is {1, 3, IN_IMG_HEIGHT, IN_IMG_WIDTH}
    // perform argmax along axis 1 (channels)
    cv::Mat cv_img = cv::Mat::zeros(IN_IMG_WIDTH, IN_IMG_HEIGHT, CV_8UC4);
    for (int i = 0; i < IN_IMG_HEIGHT; ++i)
    {
        for (int j = 0; j < IN_IMG_WIDTH; j++)
        {
            cv_img.at<cv::Vec4b>(i, j) = cv::Vec4b(raylib_image.at(i * kRowWidth + j * 4 + 2),
                                                   raylib_image.at(i * kRowWidth + j * 4 + 1),
                                                   raylib_image.at(i * kRowWidth + j * 4),
                                                   raylib_image.at(i * kRowWidth + j * 4 + 3));
        }
    }

    // cv::imshow("cv_img", cv_img);
    // cv::waitKey(10);
    return cv_img;
}

// Converts the image to the format expected by the network
void convertAndNormalizeImg(cv::Mat &img)
{
    // Resize the image to the size expected by the model.
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(INFER_HEIGHT, INFER_WIDTH));

    // Normalize the image pixels to be in the range [0, 1].
    img.convertTo(img, CV_32FC3);
    img /= 255.0;
}

// Creates ONNX Runtime input tensor
void generateFlatTensorData(const cv::Mat &img, std::vector<float> &flat_input_vec)
{
    // This model expects NCHW format (Batch, Channels, INFER_HEIGHT, INFER_WIDTH) for input serialization
    size_t input_tensor_size = BATCH_SIZE * IN_CHANNEL_SIZE * INFER_HEIGHT * INFER_WIDTH;
    flat_input_vec.resize(input_tensor_size);
    float *input_tensor_data = flat_input_vec.data();
    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < INFER_HEIGHT; ++h)
        {
            for (int w = 0; w < INFER_WIDTH; ++w)
            {
                input_tensor_data[c * INFER_HEIGHT * INFER_WIDTH + h * INFER_WIDTH + w] = img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

// Given the tensor output pointer, uses argmax to get the highest class and generate segmented image
void convertOutputAndShow(const float *output_array)
{
    // assuming output shape is {1, 3, INFER_HEIGHT, INFER_WIDTH}
    // perform argmax along axis 1 (channels)
    cv::Mat seg = cv::Mat::zeros(INFER_HEIGHT, INFER_WIDTH, CV_8UC3);
    for (int i = 0; i < INFER_HEIGHT; ++i)
    {
        for (int j = 0; j < INFER_WIDTH; ++j)
        {
            int   max_id  = 0;
            float max_val = output_array[i * INFER_WIDTH + j];
            for (int c = 1; c < OUT_CHANNEL_SIZE; ++c)
            {
                float val = output_array[c * INFER_HEIGHT * INFER_WIDTH + i * INFER_WIDTH + j];
                if (val > max_val)
                {
                    max_val = val;
                    max_id  = c;
                }
            }

            // Assign color class based on highest probably class
            if (max_id == 1)
                seg.at<cv::Vec3b>(i, j) = cv::Vec3b(250, 0, 0);
            else if (max_id == 2)
                seg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 250);
            else if (max_id == 3)
                seg.at<cv::Vec3b>(i, j) = cv::Vec3b(128, 128, 128);
        }
    }

    cv::imshow("seg", seg);
    cv::waitKey(10);
}

int main()
{
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

    const std::string model_path = "semseg.onnx";
    Ort::Session      onnx_session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo   memInfo =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape{BATCH_SIZE, IN_CHANNEL_SIZE, INFER_HEIGHT, INFER_WIDTH};

    const char *input_names[]  = {onnx_session.GetInputName(0, Ort::AllocatorWithDefaultOptions())};
    const char *output_names[] = {onnx_session.GetOutputName(0, Ort::AllocatorWithDefaultOptions())};

    // Create shared mem reader
    Q<okitch::SharedMsg<IN_IMG_WIDTH, IN_IMG_HEIGHT>, 4> *q; // shared memory object
    q = shmmap<okitch::SharedMsg<IN_IMG_WIDTH, IN_IMG_HEIGHT>, 4>("raylib_semseg_input_shmem");
    assert(q);

    auto reader = q->getReader();
    std::cout << "reader size: " << sizeof(reader) << std::endl;

    while (true)
    {
        okitch::SharedMsg<IN_IMG_WIDTH, IN_IMG_HEIGHT> *msg = reader.readLast();
        if (!msg)
        {
            continue;
        }
        auto img = convertRaylibToOpencv(msg->data);

        convertAndNormalizeImg(img);

        std::vector<float> flat_input_vec;
        generateFlatTensorData(img, flat_input_vec);

        // Create input tensor object from flattened data
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memInfo, flat_input_vec.data(), flat_input_vec.size(), input_shape.data(), input_shape.size());

        // Run the model on the input tensor and collect outputs.
        auto                    t0 = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> output_tensors =
            onnx_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        {
            auto infer_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "inference: " << infer_dur << " ms." << std::endl;
        }

        // Get pointer to output tensor float data.
        float *output_array = output_tensors[0].GetTensorMutableData<float>();

        convertOutputAndShow(output_array);
    }

    return 0;
}
