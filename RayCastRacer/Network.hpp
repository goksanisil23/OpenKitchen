#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "raylib_msgs.h"

namespace evo_driver
{

float normalizeAngleDeg(float angle)
{
    while (angle < 360.F)
    {
        angle += 360.F;
    }
    while (angle >= 360.F)
    {
        angle -= 360.F;
    }
    return angle;
}

void writeMatrixToFile(const std::string &filename, const Eigen::MatrixXf &matrix)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << matrix.rows() << " " << matrix.cols() << std::endl;
        for (int i = 0; i < matrix.rows(); i++)
        {
            for (int j = 0; j < matrix.cols(); j++)
            {
                file << matrix(i, j);
                if (j < matrix.cols() - 1)
                    file << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
}

void readMatrixFromFile(const std::string &filename, Eigen::MatrixXf &matrix)
{
    int rows, cols;

    std::ifstream file(filename);
    if (file.is_open())
    {
        file >> rows;
        file >> cols;
        matrix = Eigen::MatrixXf(rows, cols);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                double item = 0.0;
                file >> item;
                matrix(row, col) = item;
            }
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
        return;
    }
}

// Inputs: Speed, rotation (w.r.t world), measured distances from sensor
// Outputs: Acceleration, deceleration, turn_left (soft + hard), turn_right (soft + hard)
class Network
{
  public:
    enum class ClassificationLayer
    {
        Sigmoid,
        Softmax
    };

    static constexpr uint16_t            kInputsFromSensor{15};
    static constexpr uint16_t            kInputsSize{kInputsFromSensor + 2};
    static constexpr uint16_t            kOutputSize{6};
    static constexpr uint16_t            kHiddenLayerSize{30}; // 100
    static constexpr ClassificationLayer kClassificationLayerType{ClassificationLayer::Sigmoid};

    Network()
    {
        // Initialize with random weights in [-1,1] range
        weights_1_ = Eigen::Matrix<float, kInputsSize, kHiddenLayerSize>::Random();
        weights_2_ = Eigen::Matrix<float, kHiddenLayerSize, kOutputSize>::Random();

        // Eigen::MatrixXf w1, w2;
        // readMatrixFromFile("agent_weights_1.txt", w1);
        // readMatrixFromFile("agent_weights_2.txt", w2);

        // weights_1_ = w1;
        // weights_2_ = w2;
    }

    std::array<float, kOutputSize>
    infer(const float speed, const float rotation, const std::vector<okitch::Vec2d> &sensor_hits)
    {
        constexpr float kSpeedLimit{100.F};
        constexpr float kRotationLimit{360.F};
        constexpr float kSensorRange{200.F};
        // populate the input, but normalize the features so that all lie in [0,1]
        // inputs_buffer_(0) = (speed + kSpeedLimit) / (2.F * kSpeedLimit);
        inputs_buffer_(0) = speed / kSpeedLimit; // since we don't allow (-) speed
        inputs_buffer_(1) = normalizeAngleDeg(rotation) / kRotationLimit;
        for (size_t i{0}; i < sensor_hits.size(); i++)
        {
            inputs_buffer_(2 + i) = sensor_hits[i].norm() / kSensorRange;
        }

        // Compute hidden layer
        hidden_layer_buffer_ = relu(inputs_buffer_ * weights_1_);

        // Compute output layer
        if constexpr (kClassificationLayerType == ClassificationLayer::Sigmoid)
        {
            outputs_buffer_ = sigmoid(hidden_layer_buffer_ * weights_2_);
        }
        else if constexpr (kClassificationLayerType == ClassificationLayer::Softmax)
        {
            outputs_buffer_ = softmax(hidden_layer_buffer_ * weights_2_);
        }
        else
        {
            assert(false && "Classification layer type unsupported");
        }

        // Copy the output
        std::array<float, kOutputSize> out_array;
        Eigen::Map<Eigen::VectorXf>(out_array.data(), out_array.size()) = outputs_buffer_;
        return out_array;
    }

    static Eigen::VectorXf relu(const Eigen::VectorXf &input)
    {
        return input.array().max(0.F);
    }

    static Eigen::VectorXf sigmoid(const Eigen::VectorXf &input)
    {
        return (1.F / (1.F + ((-input.array()).exp())));
    }

    static Eigen::VectorXf softmax(const Eigen::VectorXf &input)
    {
        // We normalize the input first to avoid numerical instabilities and overflows
        Eigen::VectorXf expoVec{(input.array() - input.array().maxCoeff()).exp()};
        return expoVec / expoVec.sum();
    }

    Eigen::Matrix<float, kInputsSize, kHiddenLayerSize> weights_1_;
    Eigen::Matrix<float, kHiddenLayerSize, kOutputSize> weights_2_;

    Eigen::Matrix<float, 1, kInputsSize>      inputs_buffer_;
    Eigen::Matrix<float, 1, kOutputSize>      outputs_buffer_;
    Eigen::Matrix<float, 1, kHiddenLayerSize> hidden_layer_buffer_;
};

} // namespace evo_driver