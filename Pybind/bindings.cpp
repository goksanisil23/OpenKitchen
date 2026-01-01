#include "Environment/Environment.h"
#include "Environment/ScreenGrabber.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PythonControlledAgent : public Agent
{
  public:
    using Agent::Agent;
    void updateAction() override
    {
        // No-op: action will be set from Python
    }
};

class BoundEnv
{
  public:
    BoundEnv(const std::string &race_track_path, const bool draw_rays = true, const bool hidden_window = false)
    {
        agents_.push_back(std::make_unique<PythonControlledAgent>(Vec2d{0, 0}, 0, 0));
        env_ = std::make_unique<Environment>(race_track_path, createBaseAgentPtrs(agents_), draw_rays, hidden_window);

        // Place the agent at the starting position
        // const auto start_idx = RaceTrack::kStartingIdx;
        const auto  start_idx = env_->pickRandomResetTrackIdx();
        const float start_x   = env_->race_track_->track_data_points_.x_m[start_idx];
        const float start_y   = env_->race_track_->track_data_points_.y_m[start_idx];
        agents_[0]->reset({start_x, start_y}, env_->race_track_->headings_[start_idx]);

        env_->visualizer_->setAgentToFollow(agents_[0].get());
    }

    void setAction(const float throttle_delta, const float steering_delta)
    {
        agents_[0]->current_action_.throttle_delta = throttle_delta;
        agents_[0]->current_action_.steering_delta = steering_delta;
    }

    void step()
    {
        env_->step();
    }

    std::vector<uint8_t> getRenderTargetHost()
    {
        return env_->getRenderTargetHost();
    }
    ScreenGrabber::RenderTargetInfo getRenderTargetInfo()
    {
        return env_->getRenderTargetInfo();
    }

  private:
    std::unique_ptr<Environment>                        env_;
    std::vector<std::unique_ptr<PythonControlledAgent>> agents_;
};

PYBIND11_MODULE(open_kitchen_pybind, m)
{
    py::class_<ScreenGrabber::RenderTargetInfo>(m, "RenderTargetInfo")
        .def_readonly("width", &ScreenGrabber::RenderTargetInfo::width)
        .def_readonly("height", &ScreenGrabber::RenderTargetInfo::height)
        .def_readonly("channels", &ScreenGrabber::RenderTargetInfo::channels)
        .def("row_bytes", &ScreenGrabber::RenderTargetInfo::row_bytes);

    py::class_<BoundEnv>(m, "Environment")
        .def(py::init<const std::string &, bool, bool>(),
             py::arg("race_track_path"),
             py::arg("draw_rays")     = true,
             py::arg("hidden_window") = true)
        .def("set_action", &BoundEnv::setAction)
        .def("step", &BoundEnv::step)
        .def("get_render_target", &BoundEnv::getRenderTargetHost)
        .def("get_render_target_info", &BoundEnv::getRenderTargetInfo);
}
