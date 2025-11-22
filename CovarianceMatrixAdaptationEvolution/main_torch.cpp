#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "CmaEsSolverTorch.h"
#include "Controller.h"
#include "Environment/Environment.h"

const int      kPopulationSize = 20;
constexpr bool kResetAgentsRandomly{false};

class CmaEsAgent : public Agent
{
  public:
    static constexpr int64_t kHiddenSize = 16; // 16 hidden units
    static constexpr int64_t kOutputSize = 1;  // 1 output (steering)

    CmaEsAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        device_ = torch::Device(torch::kCPU);

        // Re-configure the sensor ray angles so that we only have 5 rays
        constexpr int   num_rays     = 128;
        constexpr float kRayMinAngle = -70.F;
        constexpr float kRayMaxAngle = 70.F;
        sensor_ray_angles_.clear();
        for (int i{0}; i < num_rays; i++)
        {
            float angle = kRayMinAngle + i * (kRayMaxAngle - kRayMinAngle) / (num_rays - 1);
            sensor_ray_angles_.push_back(angle);
        }

        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;

        // 5 inputs, 16 hidden, 1 output
        controller_ =
            std::make_unique<Controller>(static_cast<int64_t>(sensor_ray_angles_.size()), kHiddenSize, kOutputSize);
        controller_->to(device_);
    }

    // Create an input tensor to the network from the ego-states and sensor measurement
    torch::Tensor stateToTensor()
    {
        std::vector<float> sensor_hits_norm(sensor_hits_.size());
        for (size_t i{0}; i < sensor_hits_.size(); i++)
        {
            sensor_hits_norm[i] = sensor_hits_[i].norm() / kSensorRange;
        }
        auto cpu_view = torch::from_blob(sensor_hits_norm.data(),
                                         {static_cast<int64_t>(sensor_hits_norm.size())},
                                         torch::TensorOptions().dtype(torch::kFloat32));

        return cpu_view.clone().to(device_);
    }

    void updateAction() override
    {
        torch::NoGradGuard no_grad;
        current_state_tensor_       = stateToTensor();
        torch::Tensor action_tensor = controller_->forward(current_state_tensor_);

        // Action tensor is [-1, 1]. Scale it to a reasonable range.
        // current_action_.throttle_delta = (action_tensor[0].item<float>() + 1.0F) / 2.0F * 100.F; // Scale to [0, 100]
        current_action_.throttle_delta = 100.F;
        current_action_.steering_delta = action_tensor[0].item<float>() * 5.0F; // Scale to [-5, 5]
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void reset(const Vec2d &reset_pos, const float reset_rot) override
    {
        Agent::reset(reset_pos, reset_rot);
        fitness_ = 0.F;
    }

  public:
    torch::Tensor               current_state_tensor_;
    std::unique_ptr<Controller> controller_;
    torch::Device               device_{torch::kCPU};
    float                       fitness_{0.F};

    size_t prev_track_idx_{};
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

    const int   num_params = agents[0]->controller_->count_params();
    CmaEsSolver cma_solver(num_params, kPopulationSize);
    std::cout << "Number of parameters to optimize: " << num_params << std::endl;

    Environment env(argv[1], createBaseAgentPtrs(agents));

    const float start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    auto       &agent = agents[0];
    agent->reset({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx]);

    uint32_t episode_idx{0};

    while (true)
    {
        // Ask the solver for a new population of candidate parameters
        std::vector<torch::Tensor>      population = cma_solver.sample();
        std::vector<SolutionAndFitness> solution_fitness_pairs;
        solution_fitness_pairs.resize(kPopulationSize);

        for (int i = 0; i < kPopulationSize; i++)
        {
            env.resetAgent(agent.get(), kResetAgentsRandomly);
            agent->controller_->set_params(population[i]);

            // need to get an initial observation for the intial action, after reset
            env.step();
            agent->prev_track_idx_ = env.race_track_->findNearestTrackIndexBruteForce(agent->pos_);

            while (!agent->crashed_)
            {
                agent->updateAction();
                env.step();
                if (!agent->crashed_)
                {
                    const size_t  curr_track_idx = env.race_track_->findNearestTrackIndexBruteForce(agent->pos_);
                    const int32_t progress{static_cast<int32_t>(curr_track_idx) -
                                           static_cast<int32_t>(agent->prev_track_idx_)};
                    agent->prev_track_idx_ = curr_track_idx;
                    agent->fitness_ += static_cast<float>(std::abs(progress));
                }
                else if (agent->timed_out_)
                {
                    agent->fitness_ = 0.F;
                }
            }
            solution_fitness_pairs.push_back({population[i], agent->fitness_});
        }

        cma_solver.tell(solution_fitness_pairs);

        double best_fitness = 0;
        for (const auto &pair : solution_fitness_pairs)
        {
            if (pair.fitness > best_fitness)
            {
                best_fitness = pair.fitness;
            }
        }
        std::cout << "Generation " << episode_idx << " Best Fitness: " << best_fitness << std::endl;

        episode_idx++;
    }

    return 0;
}