#include <cassert>
#include <chrono>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

#include "raylib_img_msg.h"
#include "spmc_queue.h"

using namespace std;
constexpr int WIDTH  = 1000;
constexpr int HEIGHT = 1000;

// Converts the serialized raylib image to opencv for visualization
void convertRaylibToOpencv(const std::array<uint8_t, WIDTH * HEIGHT * 4> &raylib_image)
{
    constexpr int kRowWidth = WIDTH * 4;
    // assuming output shape is {1, 3, height, width}
    // perform argmax along axis 1 (channels)
    cv::Mat seg = cv::Mat::zeros(WIDTH, HEIGHT, CV_8UC4);
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            seg.at<cv::Vec4b>(i, j) = cv::Vec4b(raylib_image.at(i * kRowWidth + j * 4 + 2),
                                                raylib_image.at(i * kRowWidth + j * 4 + 1),
                                                raylib_image.at(i * kRowWidth + j * 4),
                                                raylib_image.at(i * kRowWidth + j * 4 + 3));
        }
    }

    cv::imshow("seg", seg);
    cv::waitKey(10);
}

// use taskset -c to bind core
int main(int argc, char **argv)
{

    Q<okitch::SharedMsg<WIDTH, HEIGHT>, 4> *q; // shared memory object
    q = shmmap<okitch::SharedMsg<WIDTH, HEIGHT>, 4>(okitch::shm_file_semseg_in);
    assert(q);

    auto reader = q->getReader();
    cout << "reader size: " << sizeof(reader) << endl;

    while (true)
    {
        okitch::SharedMsg<WIDTH, HEIGHT> *msg = reader.readLast();
        if (!msg)
        {
            // std::cout << "No msg yet..." << std::endl;
            continue;
        }
        // cout << "i: " << msg->idx << std::endl;
        // cout << "data: " << (unsigned)(msg->data[3]) << std::endl;
        convertRaylibToOpencv(msg->data);
    }

    return 0;
}
