#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>

// Define a simple struct for convenience
struct Sample
{
    torch::Tensor state;  // size [7]
    torch::Tensor action; // size [2]
};

std::vector<Sample> random_sample(const std::vector<Sample> &dataset, size_t N)
{
    static std::random_device rd;
    static std::mt19937       gen(rd());

    std::vector<Sample> sampled;
    std::sample(dataset.begin(), dataset.end(), std::back_inserter(sampled), N, gen);

    return sampled;
}

// Load all data samples from txt files in a folder
std::vector<Sample> load_dataset(const std::string &data_folder)
{
    namespace fs = std::filesystem;

    constexpr int kNumRaysToRead{7};

    // set by kSteeringAngleClampDeg in collect_data_main.cpp
    constexpr float kMaxSteeringMag{10.F};
    constexpr float kMaxThrottleMag{100.F};
    constexpr float kMaxLaserRange{200.F}; // set by kSensorRange in Agent.
    constexpr float kMaxLaserRangeSquared{kMaxLaserRange * kMaxLaserRange};

    std::vector<Sample> dataset;

    for (const auto &entry : fs::directory_iterator(data_folder))
    {
        if (entry.path().extension() == ".txt")
        {
            std::ifstream      infile(entry.path());
            std::string        line;
            std::vector<float> state_vals;

            // Read first 7 lines (laser rays)
            for (int i = 0; i < kNumRaysToRead && std::getline(infile, line); ++i)
            {
                std::istringstream iss(line);
                float              x, y;
                iss >> x >> y;
                // Normalize to [0, 1]
                state_vals.push_back((x * x + y * y) / kMaxLaserRangeSquared);
            }

            torch::Tensor state = torch::tensor(state_vals);

            // Read last line (action)
            if (std::getline(infile, line))
            {
                std::istringstream iss(line);
                float              throttle, steering;
                iss >> throttle >> steering;
                // Normalize to throttle from [0, 100] to [-1, 1] and steering from [-10, 10] to [-1, 1]
                torch::Tensor action =
                    torch::tensor({((throttle / kMaxThrottleMag) - 0.5F) * 2.0F, steering / kMaxSteeringMag});

                dataset.push_back({state, action});
            }
        }
    }
    std::cout << "Loaded " << dataset.size() << " samples" << std::endl;

    return dataset;
}