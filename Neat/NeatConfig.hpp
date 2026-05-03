#pragma once

#include <cstdint>

namespace neat
{
struct Config
{
    static constexpr uint32_t kMaxGenerations{0U}; // 0 means run indefinitely
    static constexpr uint32_t kMaxStepsPerGeneration{4000U};
    static constexpr uint32_t kRandomSeed{42U};

    static constexpr int32_t kNumRayInputs{15};
    static constexpr int32_t kNumOutputs{2};
    static constexpr float   kSigmoidSlope{4.9F};

    static constexpr float kWeightInitMin{-1.0F};
    static constexpr float kWeightInitMax{1.0F};
    static constexpr float kThrottleDeltaLimit{0.30F};
    static constexpr float kSteeringDeltaLimitDeg{4.0F};

    static constexpr float kCompatibilityThreshold{3.0F};
    static constexpr float kExcessCoeff{1.0F};
    static constexpr float kDisjointCoeff{1.0F};
    static constexpr float kWeightDiffCoeff{0.4F};

    static constexpr float    kSurvivalFraction{0.5F};
    static constexpr uint32_t kStagnationGenerations{15U};
    static constexpr float    kCrossoverRate{0.75F};
    static constexpr float    kDisabledInheritanceRate{0.75F};

    static constexpr float kWeightMutationRate{0.80F};
    static constexpr float kWeightPerturbRate{0.90F};
    static constexpr float kWeightPerturbScale{0.5F};
    static constexpr float kAddConnectionRate{0.05F};
    static constexpr float kAddNodeRate{0.03F};
    static constexpr float kReenableConnectionRate{0.05F};
};
} // namespace neat
