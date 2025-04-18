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
    PythonInferAgent(const Vec2d start_pos, const float start_rot, const int16_t id) : Agent(start_pos, start_rot, id)
    {
        // Setup shared memory and semaphores
        int shm_fd = shm_open("myshm", O_CREAT | O_RDWR, 0666);
        ftruncate(shm_fd, 1024);
        shmem_ptr = mmap(0, 1024, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);

        sem1_ = sem_open("/sem1", O_CREAT, 0666, 0);
        sem2_ = sem_open("/sem2", O_CREAT, 0666, 0);

        // Setup sensor
        sensor_ray_angles_.clear();
        sensor_ray_angles_.push_back(-90.F);
        sensor_ray_angles_.push_back(-60.F);
        sensor_ray_angles_.push_back(-30.F);
        sensor_ray_angles_.push_back(0.F);
        sensor_ray_angles_.push_back(30.F);
        sensor_ray_angles_.push_back(60.F);
        sensor_ray_angles_.push_back(90.F);

        current_action_.throttle_delta = 0.F;
        current_action_.steering_delta = 0.F;
    }
    void updateAction()
    {
        current_action_.throttle_delta = action_response_buffer_[0];
        current_action_.steering_delta = action_response_buffer_[1];
        std::cout << "acc: " << current_action_.throttle_delta << " str: " << current_action_.steering_delta
                  << std::endl;
    }

    void sendMeasurementsReceiveAction()
    {
        // Serialize measurements
        size_t i = 0;
        for (const auto &hit : sensor_hits_)
        {
            measurement_buffer_[i]     = hit.x;
            measurement_buffer_[i + 1] = hit.y;
            i += 2;
        }

        // Send
        memcpy(shmem_ptr, measurement_buffer_, sizeof(measurement_buffer_));
        // Signal Python that data is ready
        sem_post(sem1_);
        // Wait for Python to process and respond
        sem_wait(sem2_);
        // Read response
        memcpy(action_response_buffer_,
               static_cast<char *>(shmem_ptr) + sizeof(measurement_buffer_),
               sizeof(action_response_buffer_));
    }

    void reset(const raylib::Vector2 &reset_pos, const float reset_rot, const size_t track_reset_idx)
    {
        Agent::reset(reset_pos, reset_rot);

        sensor_hits_ = std::vector<Vec2d>(sensor_ray_angles_.size());
    }

  private:
    sem_t *sem1_;
    sem_t *sem2_;
    void  *shmem_ptr;

    float measurement_buffer_[14];    // 7 rays, 2d points
    float action_response_buffer_[2]; // acceleration & steering
};

int32_t pickResetPosition(const Environment &env, const Agent *agent)
{
    return GetRandomValue(0, static_cast<int32_t>(env.race_track_->track_data_points_.x_m.size()) - 1);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a folder path for race track database\n";
        return -1;
    }

    Environment       env(argv[1]);
    const std::string track_name{env.race_track_->track_name_};

    const float      start_pos_x{env.race_track_->track_data_points_.x_m[RaceTrack::kStartingIdx]};
    const float      start_pos_y{env.race_track_->track_data_points_.y_m[RaceTrack::kStartingIdx]};
    PythonInferAgent agent({start_pos_x, start_pos_y}, env.race_track_->headings_[RaceTrack::kStartingIdx], 0);
    env.setAgent(&agent);

    int32_t reset_idx{RaceTrack::kStartingIdx};
    while (true)
    {
        agent.reset(
            {env.race_track_->track_data_points_.x_m[reset_idx], env.race_track_->track_data_points_.y_m[reset_idx]},
            env.race_track_->headings_[reset_idx],
            env.race_track_->findNearestTrackIndexBruteForce({env.race_track_->track_data_points_.x_m[reset_idx],
                                                              env.race_track_->track_data_points_.y_m[reset_idx]}));
        while (!agent.crashed_)
        {
            env.step(); // agent moves in the environment with current_action, produces next_state
            agent.sendMeasurementsReceiveAction();
            agent.updateAction();
        }
        reset_idx = pickResetPosition(env, &agent);
    }

    return 0;
}