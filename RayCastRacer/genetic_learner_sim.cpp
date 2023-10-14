#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Environment/Visualizer.h"
#include "EnvironmentUtils.hpp"

#include "raylib-cpp.hpp"
#include "spmc_queue.h"

#include "Environment/IpcMsgs.h"
#include "Environment/RaceTrack.hpp"
#include "GeneticAgent.hpp"
#include "Mating.hpp"

constexpr int16_t kNumAgents{50};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Provide a file path for trace track csv file\n";
        return -1;
    }

    RaceTrack race_track(argv[1]);

    env::Visualizer visualizer;

    std::vector<genetic::GeneticAgent> agents;
    for (int16_t i{0}; i < kNumAgents; i++)
    {
        agents.push_back(genetic::GeneticAgent({race_track.track_data_points_.x_m[RaceTrack::kStartingIdx],
                                                race_track.track_data_points_.y_m[RaceTrack::kStartingIdx]},
                                               race_track.init_heading_deg_,
                                               i));
    }

    auto sensor_msg_shm_q = shmmap<Laser2dMsg<100>, 4>("laser_msgs"); // shared memory object
    assert(sensor_msg_shm_q);

    bool reset_episode = false;

    uint32_t           episode_idx{0};
    uint32_t           iteration{0};
    std::vector<float> colony_avg_scores;
    while (!WindowShouldClose())
    {
        if (reset_episode)
        {
            std::cout << "------------ EPISODE " << episode_idx << " DONE ---------------" << std::endl;
            iteration = 0;
            episode_idx++;

            // First, assign scores based on how far along the track the agents has come
            genetic::util::assignScores(agents, race_track);
            genetic::util::saveBestAgentNetwork(agents);
            colony_avg_scores.emplace_back(genetic::util::getAvgColonyScore(agents));
            genetic::util::showColonyScore(colony_avg_scores);

            // Mate the agents before resetting
            genetic::chooseAndMateAgents(agents);

            for (auto &agent : agents)
            {
                agent.reset({race_track.track_data_points_.x_m[RaceTrack::kStartingIdx],
                             race_track.track_data_points_.y_m[RaceTrack::kStartingIdx]},
                            race_track.init_heading_deg_);
            }
            reset_episode = false;
        }

        // -------- Kinematics Update of Agents -------- //
        for (auto &agent : agents)
        {
            if (!agent.crashed_)
            {
                agent.move();
                if (agent.standstill_timed_out_)
                {
                    agent.score_   = 1; // minimum non-zero score
                    agent.crashed_ = true;
                }
            }
        }
        visualizer.activateDrawing();
        {
            env::Visualizer::shadeAreaBetweenCurves(
                race_track.right_bound_inner_, race_track.right_bound_outer_, raylib::Color(0, 0, 255, 255));
            env::Visualizer::shadeAreaBetweenCurves(
                race_track.left_bound_inner_, race_track.left_bound_outer_, raylib::Color(255, 0, 0, 255));
            env::Visualizer::shadeAreaBetweenCurves(
                race_track.left_bound_inner_, race_track.right_bound_inner_, raylib::Color(0, 255, 0, 255));
            env::Visualizer::shadeAreaBetweenCurves(
                race_track.start_line_, race_track.finish_line_, raylib::Color(0, 0, 255, 255));
            // drawTrackPointNumbers(track_data_points);
            genetic::util::drawActionBar(agents, iteration);
            genetic::util::drawEpisodeNum(episode_idx);
            genetic::util::drawTrackTitle(race_track.track_name_);
        }
        visualizer.disableDrawing();

        // This section is for direct render buffer manipulation
        {
            raylib::Image render_buffer;
            render_buffer.Load(visualizer.render_target_.texture);
            // We first check the collision before drawing any sensor or agents to avoid overlap
            // NOTE: Sensor update needs to happen before drawing multiple agents since we emulate parallel simulators here so agents
            // should NOT see each other's world.
            // for (auto &driver : agents)
            for (size_t i{0}; i < agents.size(); i++)
            {
                if (!agents[i].crashed_)
                {
                    if (visualizer.checkAgentCollision(render_buffer, agents[i]))
                    {
                        agents[i].crashed_ = true;
                    }
                    else
                    {
                        visualizer.updateSensor(agents[i], render_buffer);
                    }
                }
            }
            for (auto &agent : agents)
                visualizer.drawAgent(agent, render_buffer);
            UpdateTexture(visualizer.render_target_.texture, render_buffer.data);
        }

        // Render the final texture
        visualizer.render();

        for (auto &agent : agents)
        {
            agent.updateAction();
        }

        // If all the robots have crashed, reset the generation
        reset_episode = genetic::util::shouldResetEpisode(agents);
        iteration++;

        // Send the sensor readings over shared mem
        // sensor_msg_shm_q->write(
        //     [&sensor_hits](Laser2dMsg<100> &msg)
        //     {
        //         msg.size = sensor_hits[0].size();
        //         for (size_t i{0}; i < sensor_hits[0].size(); i++)
        //         {
        //             msg.data[i] = sensor_hits[0][i];
        //         }
        //     });
    }

    return 0;
}