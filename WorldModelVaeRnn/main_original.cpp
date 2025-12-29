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

constexpr int   kPopulationSize = 30;
constexpr bool  kResetAgentsRandomly{false};
constexpr bool  kTrainMode{false};
constexpr float kMaxDistToBound{50.F};
constexpr int   kNumEpisodes{300};

torch::Tensor loadStatVector(std::ifstream &file, const int dim)
{
    std::string line;
    if (std::getline(file, line))
    {
        std::vector<float> values;
        std::stringstream  ss(line);
        float              value;
        while (ss >> value)
        {
            values.push_back(value);
        }
        if (static_cast<int>(values.size()) != dim)
        {
            throw std::runtime_error("Stat dimension mismatch!");
        }
        return torch::tensor(values, torch::kFloat32).view({1, dim});
    }
    throw std::runtime_error("Could not read line from stats file.");
}

// Same as in train_vae.py
torch::Tensor reparameterize(const torch::Tensor &mu, const torch::Tensor &logvar)
{
    torch::Tensor std = torch::exp(0.5 * logvar);
    torch::Tensor eps = torch::randn_like(std);
    return mu + eps * std;
}

// Image Post-processing: torch::Tensor -> cv::Mat
cv::Mat tensorToImage(torch::Tensor tensor)
{
    // Note: torch stores in [RRR...GGG...BBB...] format
    // opencv requries interleaved [RGBRGBRGB...] format, hence we need to use contiguous()
    tensor = tensor.squeeze(0).clamp(0, 1).to(torch::kCPU); // CxHxW
    tensor = tensor.permute({1, 2, 0}).contiguous();        // HxWxC
    tensor = tensor.mul(255).to(torch::kByte);

    int height = tensor.size(0);
    int width  = tensor.size(1);

    cv::Mat mat(height, width, CV_8UC3, tensor.data_ptr<uchar>());
    cv::Mat mat_bgr;
    cv::cvtColor(mat, mat_bgr, cv::COLOR_RGB2BGR);
    return mat_bgr;
}

class CmaEsAgent : public Agent
{
  public:
    static constexpr int VAE_IMG_SIZE = 128; // obtained from train_vae.py
    static constexpr int Z_DIM        = 32;  // obtained from train_vae.py
    static constexpr int A_DIM        = 2;   // obtained from train_vae.py
    static constexpr int STRIDE       = 1;
    static constexpr int SEQ_LEN      = 1;

    static constexpr int64_t kInputSize  = Z_DIM + 2; //
    static constexpr int64_t kHiddenSize = 16;        // Size of hidden layer
    static constexpr int64_t kOutputSize = 1;         // 1 output (steering)

    CmaEsAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;

        setupWorldModel();

        controller_ = std::make_unique<Controller>(kInputSize, kHiddenSize, 1);
        controller_->to(device_);

        if (!kTrainMode)
        {
            std::string const parameters_path = "best_controller_params_episode_289.pt";
            torch::Tensor     t;
            torch::load(t, parameters_path);
            controller_->set_params(t.to(device_));
            std::cout << "Loaded controller parameters from " << parameters_path << std::endl;
        }
    }

    void setupWorldModel()
    {
        vae_encoder_ = torch::jit::load("../cpp_models_32/vae_encoder.pt");
        vae_decoder_ = torch::jit::load("../cpp_models_32/vae_decoder.pt");
        rnn_model_   = torch::jit::load("../cpp_models/rnn_model.pt");

        vae_encoder_.to(device_);
        vae_decoder_.to(device_);
        rnn_model_.to(device_);

        vae_encoder_.eval();
        vae_decoder_.eval();
        rnn_model_.eval();

        // // Load normalization stats
        // std::ifstream stats_file("../cpp_models/stats.txt");
        // auto          z_mean_tmp = loadStatVector(stats_file, Z_DIM).to(device_);
        // auto          z_std_tmp  = loadStatVector(stats_file, Z_DIM).to(device_);
        // auto          a_mean_tmp = loadStatVector(stats_file, A_DIM).to(device_);
        // auto          a_std_tmp  = loadStatVector(stats_file, A_DIM).to(device_);

        // z_mean_ = z_mean_tmp.view({1, 1, Z_DIM});
        // z_std_  = z_std_tmp.view({1, 1, Z_DIM});
        // a_mean_ = a_mean_tmp.view({1, 1, A_DIM});
        // a_std_  = a_std_tmp.view({1, 1, A_DIM});

        // stats_file.close();
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

        const torch::Tensor vae_input = interpolate(chw.unsqueeze(0),
                                                    InterpolateFuncOptions()
                                                        .size(std::vector<long>{VAE_IMG_SIZE, VAE_IMG_SIZE})
                                                        .mode(torch::kBilinear)
                                                        .align_corners(false)); // 1x3xSxS

        return vae_input;
    }

    void runVaeRnn(const torch::Tensor vae_input)
    {
        static size_t       ctr{0};
        const auto          encoder_output = vae_encoder_.forward({vae_input}).toTuple();
        const torch::Tensor mu_t           = encoder_output->elements()[0].toTensor();
        const torch::Tensor logvar_t       = encoder_output->elements()[1].toTensor();

        // Get latent vector 'z' and reconstruct
        const torch::Tensor z_t = reparameterize(mu_t, logvar_t);
        // const torch::Tensor recon_t = vae_decoder_.forward({z_t}).toTensor();
        const torch::Tensor recon_t = vae_decoder_.forward({mu_t}).toTensor();

        // cv::Mat vae_rec_img = tensorToImage(recon_t);
        // // cv::imwrite("vae_" + std::to_string(ctr) + ".png", vae_rec_img);
        // cv::imshow("VAE", vae_rec_img);
        // cv::waitKey(1);

        // 4. Run RNN prediction step
        // if (step % STRIDE == 0)
        {
            torch::Tensor a_t =
                torch::tensor({{current_action_.throttle_delta, current_action_.steering_delta}}, torch::kFloat32)
                    .view({1, 1, A_DIM})
                    .to(device_);

            torch::Tensor z_t_seq = z_t.view({1, 1, Z_DIM});
            torch::Tensor z_norm  = (z_t_seq - z_mean_) / z_std_;
            torch::Tensor a_norm  = (a_t - a_mean_) / a_std_;
            torch::Tensor x_t     = torch::cat({z_norm, a_norm}, -1);

            std::deque<torch::Tensor> x_buf;
            x_buf.push_back(x_t);
            if (x_buf.size() > SEQ_LEN)
            {
                x_buf.pop_front();
            }

            if (x_buf.size() == SEQ_LEN)
            {
                torch::Tensor x_seq = torch::cat(std::vector<torch::Tensor>(x_buf.begin(), x_buf.end()), 1);

                auto          rnn_output             = rnn_model_.forward({x_seq}).toTuple();
                torch::Tensor z_next_pred_normalized = rnn_output->elements()[0].toTensor(); // (1,seq_len,Z)
                torch::Tensor h_all = rnn_output->elements()[1].toTensor(); // (num_gru_layers,1,gru_hidden_dim)

                torch::Tensor z_next = z_next_pred_normalized * z_std_ + z_mean_;
                z_next               = z_next.view({1, Z_DIM});
                torch::Tensor pred   = vae_decoder_.forward({z_next}).toTensor();

                cv::Mat pred_img = tensorToImage(pred);
                // cv::imwrite("rnn_" + std::to_string(ctr++) + ".png", pred_img);
                // cv::imshow("RNN", pred_img);
                // cv::waitKey(1);

                // last step & top layer
                int64_t       num_gru_hidden = h_all.size(0);
                torch::Tensor h_top          = h_all.select(0, num_gru_hidden - 1); // (1,gru_hidden_dim)

                // concat current z_t with hidden
                current_state_tensor_ = torch::cat({z_t.view({1, Z_DIM}), h_top}, 1); // (1, Z+H)
            }
        }
    }

    void runVaeOnly(const torch::Tensor vae_input)
    {
        const auto          encoder_output = vae_encoder_.forward({vae_input}).toTuple();
        const torch::Tensor mu_t           = encoder_output->elements()[0].toTensor();
        // const torch::Tensor logvar_t       = encoder_output->elements()[1].toTensor();

        // const torch::Tensor recon_t     = vae_decoder_.forward({mu_t}).toTensor();
        // cv::Mat             vae_rec_img = tensorToImage(recon_t);
        // cv::imshow("VAE", vae_rec_img);
        // cv::waitKey(1);

        torch::Tensor heading_tensor = torch::tensor({std::sin(rot_), std::cos(rot_)},
                                                     torch::TensorOptions().dtype(torch::kFloat32).device(device_))
                                           .view({1, 2});
        current_state_tensor_ = torch::cat({mu_t, heading_tensor}, /*dim=*/1);
    }

    void updateAction(const Environment &env)
    {
        torch::NoGradGuard no_grad;
        auto const         vae_input = stateToTensor(env);
        // runVaeRnn(vae_input);
        runVaeOnly(vae_input);

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
    torch::jit::script::Module vae_encoder_, vae_decoder_, rnn_model_;
    torch::Tensor              z_mean_, z_std_, a_mean_, a_std_;
    std::deque<torch::Tensor>  x_buf_;

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
    env.visualizer_->camera_.zoom = 10.0f;

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
                    // const size_t  curr_track_idx = env.race_track_->findNearestTrackIndexBruteForce(agent->pos_);
                    // const int32_t progress{static_cast<int32_t>(curr_track_idx) -
                    //                        static_cast<int32_t>(agent->prev_track_idx_)};
                    // agent->prev_track_idx_ = curr_track_idx;
                    // agent->fitness_ += static_cast<float>(std::abs(progress));
                    const float closest_dist = env.race_track_->getDistanceToLaneCenter(agent->pos_);
                    agent->fitness_ += (1.F - closest_dist);
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