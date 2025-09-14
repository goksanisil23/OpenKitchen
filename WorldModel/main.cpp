#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <semaphore.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "Environment/Environment.h"

// Agent that listens to a python process doing the inference
// Sends the measurements over shared memory and receives the actions back
class PythonInferAgent : public Agent
{
  public:
    static constexpr size_t kSharedMemSizeBytesMeasurement{1600 * 1400 * 4};
    static constexpr size_t kSharedMemSizeBytesActions{4 * 2};

    PythonInferAgent(const Vec2d start_pos, const float start_rot, const int16_t id)
        : Agent(start_pos, start_rot, id), measurement_buffer_(kSharedMemSizeBytesMeasurement, 0)
    {
        // Setup shared memory and semaphores
        int shm_meas_fd   = shm_open("shm_measurements", O_CREAT | O_RDWR, 0666);
        int shm_action_fd = shm_open("shm_actions", O_CREAT | O_RDWR, 0666);
        ftruncate(shm_meas_fd, kSharedMemSizeBytesMeasurement);
        ftruncate(shm_action_fd, kSharedMemSizeBytesMeasurement);
        shmem_measurements_ptr =
            mmap(0, kSharedMemSizeBytesMeasurement, PROT_WRITE | PROT_READ, MAP_SHARED, shm_meas_fd, 0);
        shmem_actions_ptr = mmap(0, kSharedMemSizeBytesActions, PROT_WRITE | PROT_READ, MAP_SHARED, shm_action_fd, 0);

        assert(shmem_measurements_ptr != MAP_FAILED);
        assert(shmem_actions_ptr != MAP_FAILED);

        sem_measurements_ = sem_open("/sem1", O_CREAT, 0666, 0);
        sem_actions_      = sem_open("/sem2", O_CREAT, 0666, 0);

        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;
    }
    void updateAction()
    {
        // Denormalize actions in [-1,1] from the neural network
        current_action_.throttle_delta = action_response_buffer_[0];
        current_action_.steering_delta = action_response_buffer_[1];
        std::cout << "acc: " << current_action_.throttle_delta << " str: " << current_action_.steering_delta
                  << std::endl;
    }

    void sendMeasurementsReceiveAction(Environment &env)
    {

        if (env.visualizer_->agent_to_follow_) // Use in map-view mode
        {
            Image const img{raylib::Image::LoadFromScreen()};
            std::memcpy(shmem_measurements_ptr, img.data, kSharedMemSizeBytesMeasurement);
            UnloadImage(img);
        }
        else // Use in follow-agent mode
        {
            std::memcpy(shmem_measurements_ptr, env.render_buffer_->data, kSharedMemSizeBytesMeasurement);
        }

        // Signal Python that data is ready
        sem_post(sem_measurements_);
        // Wait for Python to process and respond
        sem_wait(sem_actions_);
        // Read response
        memcpy(action_response_buffer_, shmem_actions_ptr, sizeof(action_response_buffer_));
    }

  private:
    sem_t *sem_measurements_;
    sem_t *sem_actions_;
    void  *shmem_measurements_ptr;
    void  *shmem_actions_ptr;

    std::vector<uint8_t> measurement_buffer_;
    float                action_response_buffer_[2]; // acceleration & steering
};

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cerr << "Provide a folder path for race track database\n";
        return -1;
    }

    std::vector<std::unique_ptr<PythonInferAgent>> agents;
    agents.push_back(std::make_unique<PythonInferAgent>(Vec2d{0, 0}, 0, 0));
    auto const &agent = agents[0];

    constexpr bool kDrawRays{false};
    Environment    env(std::string(argv[1]), createBaseAgentPtrs(agents), kDrawRays);

    env.visualizer_->setAgentToFollow(agent.get());
    env.visualizer_->camera_.zoom = 10.0f;

    while (true)
    {
        env.resetAgent(agent.get());
        while (!agent->crashed_)
        {
            agent->sendMeasurementsReceiveAction(env);
            agent->updateAction();
            env.step(); // agent moves in the environment with current_action, produces next_state
        }
    }

    return 0;
}