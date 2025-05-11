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

std::vector<Sample> randomSample(const std::vector<Sample> &dataset, size_t N)
{
    static std::random_device rd;
    static std::mt19937       gen(rd());

    std::vector<Sample> sampled;
    std::sample(dataset.begin(), dataset.end(), std::back_inserter(sampled), N, gen);

    return sampled;
}

Sample sampleRandomExpertTrajectory(const std::vector<Sample> &dataset, const size_t N)
{
    static std::random_device rd;
    static std::mt19937       gen(rd());

    std::vector<torch::Tensor> states_vec, actions_vec;

    std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);
    for (size_t i = 0; i < N; ++i)
    {
        // Sample a random index
        size_t idx = dist(gen);
        states_vec.push_back(dataset[idx].state);
        actions_vec.push_back(dataset[idx].action);
    }

    Sample sampled;
    sampled.state  = torch::stack(states_vec);
    sampled.action = torch::stack(actions_vec);
    return sampled;
}

Sample getAllExpertSamplesInDataset(const std::vector<Sample> &dataset)
{
    std::vector<torch::Tensor> states_vec, actions_vec;

    for (const auto &sample : dataset)
    {
        states_vec.push_back(sample.state);
        actions_vec.push_back(sample.action);
    }

    Sample sampled;
    sampled.state  = torch::stack(states_vec);
    sampled.action = torch::stack(actions_vec);
    return sampled;
}

// Load all data samples from txt files in a folder
std::vector<Sample> loadDataset(const std::string &data_folder, const std::string track_name)
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
            // Skip the files that doesn't contain the track_name
            if ((track_name != "") && (entry.path().string().find(track_name) == std::string::npos))
            {
                continue;
            }
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