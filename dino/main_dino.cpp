#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include "CovarianceMatrixAdaptationEvolution/CmaEsSolverTorch.h"
#include "CovarianceMatrixAdaptationEvolution/Controller.h"
#include "Environment/Environment.h"
#include "debug_viz_dino.hpp"

constexpr int   kPopulationSize = 50;
constexpr bool  kResetAgentsRandomly{false};
constexpr bool  kTrainMode{true};
constexpr float kMaxDistToBound{50.F};
constexpr int   kNumEpisodes{300};

class CmaEsAgent : public Agent
{
  public:
    static constexpr int DINO_IMG_SIZE = 224;
    static constexpr int Z_DIM         = 384;

    static constexpr int64_t kHiddenSize = 16; // Size of hidden layer
    static constexpr int64_t kOutputSize = 1;  // 1 output (steering)

    CmaEsAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;

        setupWorldModel();

        controller_ = std::make_unique<Controller>(Z_DIM, kHiddenSize, kOutputSize);
        controller_->to(device_);

        if (!kTrainMode)
        {
            std::string const parameters_path = "best_controller_params_episode_55.pt";
            torch::Tensor     t;
            torch::load(t, parameters_path);
            controller_->set_params(t.to(device_));
            std::cout << "Loaded controller parameters from " << parameters_path << std::endl;
        }
    }

    void setupWorldModel()
    {
        dino_model_ = torch::jit::load("../../dino/dino_vits16_ts.pt");
        dino_model_.to(device_);
        dino_model_.eval();
    }

    torch::Tensor stateToTensor(const Environment &env)
    {
        using namespace torch::nn::functional;
        using namespace torch::indexing;

        const auto render_target_info = env.screen_grabber_->getRenderTargetInfo();
        // HxWx4 uint8 on CUDA
        auto t = torch::empty({render_target_info.height, render_target_info.width, render_target_info.channels},
                              torch::TensorOptions().dtype(torch::kUInt8).device(device_));
        const size_t dst_pitch = static_cast<size_t>(t.stride(0)) * t.itemsize(); // width*4
        env.screen_grabber_->getRenderTargetDevice(t.data_ptr(), dst_pitch);

        // GPU transforms: flip, drop alpha, to float, NCHW, resize
        auto flipped = t.flip({0});                                           // HxWx4
        auto rgb     = flipped.index({Slice(), Slice(), Slice(0, 3)});        // HxWx3 (RGBA->RGB)
        auto chw     = rgb.permute({2, 0, 1}).to(torch::kFloat).div_(255.0f); // 3xHxW

        // Apply normalization for dino
        auto const mean_t =
            torch::tensor({0.485f, 0.456f, 0.406f}, torch::TensorOptions().dtype(torch::kFloat).device(device_))
                .view({3, 1, 1});
        auto const std_t =
            torch::tensor({0.229f, 0.224f, 0.225f}, torch::TensorOptions().dtype(torch::kFloat).device(device_))
                .view({3, 1, 1});
        chw = (chw - mean_t) / std_t;

        const torch::Tensor dino_input = interpolate(chw.unsqueeze(0),
                                                     InterpolateFuncOptions()
                                                         .size(std::vector<long>{DINO_IMG_SIZE, DINO_IMG_SIZE})
                                                         .mode(torch::kBilinear)
                                                         .align_corners(false)); // 1x3xSxS

        return dino_input;
    }

    void runDino(const torch::Tensor dino_input)
    {
        const auto dino_output = dino_model_.forward({dino_input}).toTensor();
        current_state_tensor_  = dino_output;
    }

    void runDinoPatchAverage(const torch::Tensor dino_input)
    {
        using namespace torch::indexing;

        torch::Tensor out = dino_model_.run_method("forward_features", dino_input).toTensor();
        // Drop the CLS token
        auto const patches    = out.index({Slice(), Slice(1, None), Slice()}); // [1,196,384]
        auto       z          = patches.mean(1);                               // [1,384]
        z                     = z / (z.norm(2, 1, true) + 1e-6);               // L2 normalize
        current_state_tensor_ = z;
    }

    void updateAction(const Environment &env)
    {
        torch::NoGradGuard no_grad;
        auto const         dino_input = stateToTensor(env);
        // showDinoPatchHeatmap(dino_model_, dino_input);
        // runDino(dino_input);
        runDinoPatchAverage(dino_input);

        updateAction();
    }

    void updateAction() override
    {
        current_action_.throttle_delta = 100.F;

        torch::Tensor action_tensor    = controller_->forward(current_state_tensor_);
        current_action_.steering_delta = action_tensor[0].item<float>() * 5.0F; // Scale to [-5, 5]
    }

    void reset(const Vec2d &reset_pos, const float reset_rot) override
    {
        Agent::reset(reset_pos, reset_rot);
        fitness_ = 0.F;
    }

  public:
    const torch::Device device_{torch::kCUDA};
    size_t              prev_track_idx_{};
    torch::Tensor       current_state_tensor_;

    // World model related
    torch::jit::script::Module dino_model_;

    // CMA-ES related
    float                       fitness_{0.F};
    std::unique_ptr<Controller> controller_;
};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    std::vector<std::unique_ptr<CmaEsAgent>> agents;
    // We're gonna reuse the same agent for all the population (in serial)
    agents.push_back(std::make_unique<CmaEsAgent>(Vec2d{0, 0}, 0, 0));

    constexpr bool kDrawRays{false};
    constexpr bool kHideWindow{kTrainMode};
    Environment    env(argv[1], createBaseAgentPtrs(agents), kDrawRays, kHideWindow);

    env.visualizer_->setAgentToFollow(agents[0].get());

    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    auto       &agent = agents[0];
    agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);

    const int num_params = agent->controller_->count_params();
    std::cout << "Number of parameters to optimize: " << num_params << std::endl;
    constexpr float kSigma = 0.5;
    CmaEsSolver     cma_solver(num_params, kPopulationSize, torch::kCUDA, kSigma);

    uint32_t episode_idx{0};

    while (episode_idx < kNumEpisodes)
    {
        // Ask the solver for a new population of candidate parameters
        std::vector<torch::Tensor>      population = cma_solver.sample();
        std::vector<SolutionAndFitness> solution_fitness_pairs;
        solution_fitness_pairs.reserve(kPopulationSize);

        for (int i = 0; i < kPopulationSize; i++)
        {
            env.resetAgent(agent.get(), kResetAgentsRandomly);
            if (kTrainMode)
                agent->controller_->set_params(population[i]);

            // need to get an initial observation for the intial action, after reset
            env.step();
            agent->prev_track_idx_ = env.race_track_->findNearestTrackIndexBruteForce(agent->pos_);

            while (!agent->crashed_)
            {
                agent->updateAction(env);
                env.step();
                if (!kTrainMode)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (!agent->crashed_)
                {
                    const size_t  curr_track_idx = env.race_track_->findNearestTrackIndexBruteForce(agent->pos_);
                    const int32_t progress{static_cast<int32_t>(curr_track_idx) -
                                           static_cast<int32_t>(agent->prev_track_idx_)};
                    agent->prev_track_idx_ = curr_track_idx;
                    agent->fitness_ += static_cast<float>(std::abs(progress));
                    // const float closest_dist = env.race_track_->getDistanceToLaneCenter(agent->pos_);
                    // agent->fitness_ += (1.F - closest_dist);
                    // agent->fitness_ += 1.F;
                }
                else if (agent->timed_out_)
                {
                    agent->fitness_ = 0.F;
                }
            }
            solution_fitness_pairs.push_back({population[i], agent->fitness_});
        }

        if (kTrainMode)
        {
            cma_solver.tell(solution_fitness_pairs);

            double   best_fitness = 0;
            uint32_t best_idx     = 0U;
            for (uint32_t i{0U}; i < solution_fitness_pairs.size(); ++i)
            {
                const auto &pair = solution_fitness_pairs[i];
                if (pair.fitness > best_fitness)
                {
                    best_fitness = pair.fitness;
                    best_idx     = i;
                }
            }
            std::cout << "Generation " << episode_idx << " Best Fitness: " << best_fitness << std::endl;
            {
                auto const best_params = solution_fitness_pairs[best_idx].solution;
                torch::save(best_params, "best_controller_params_episode_" + std::to_string(episode_idx) + ".pt");
            }
        }
        episode_idx++;
    }

    return 0;
}
