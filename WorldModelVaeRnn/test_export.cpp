#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <vector>

namespace fs = std::filesystem;

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

// Image Pre-processing: cv::Mat -> torch::Tensor
torch::Tensor imageToTensor(const cv::Mat &image, const int img_size, const torch::Device &device)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(img_size, img_size));

    cv::Mat image_rgb;
    cv::cvtColor(resized_image, image_rgb, cv::COLOR_BGR2RGB);

    torch::Tensor tensor_image = torch::from_blob(image_rgb.data, {img_size, img_size, 3}, torch::kByte);
    tensor_image               = tensor_image.to(torch::kFloat).div(255.0); // HxWxC
    tensor_image               = tensor_image.permute({2, 0, 1});           // CxHxW
    tensor_image               = tensor_image.unsqueeze(0);                 // 1xCxHxW
    return tensor_image.to(device);
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

int main()
{
    torch::Device device = torch::kCUDA;

    const int VAE_IMG_SIZE = 128; // obtained from train_vae.py
    const int Z_DIM        = 128; // obtained from train_vae.py
    const int A_DIM        = 2;   // obtained from train_vae.py
    const int STRIDE       = 1;
    const int SEQ_LEN      = 1;

    // ----- Load Models  -----
    torch::jit::script::Module vae_encoder, vae_decoder, rnn_model;

    vae_encoder = torch::jit::load("../cpp_models/vae_encoder.pt");
    vae_decoder = torch::jit::load("../cpp_models/vae_decoder.pt");
    rnn_model   = torch::jit::load("../cpp_models/rnn_model.pt");

    vae_encoder.to(device);
    vae_decoder.to(device);
    rnn_model.to(device);

    vae_encoder.eval();
    vae_decoder.eval();
    rnn_model.eval();

    // ----- Load Normalization Stats -----
    torch::Tensor z_mean, z_std, a_mean, a_std;
    try
    {
        std::ifstream stats_file("../cpp_models/stats.txt");
        z_mean = loadStatVector(stats_file, Z_DIM).to(device);
        z_std  = loadStatVector(stats_file, Z_DIM).to(device);
        a_mean = loadStatVector(stats_file, A_DIM).to(device);
        a_std  = loadStatVector(stats_file, A_DIM).to(device);
        stats_file.close();
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Error loading stats: " << e.what() << std::endl;
        return -1;
    }

    torch::Tensor z_mean_b = z_mean.view({1, 1, Z_DIM});
    torch::Tensor z_std_b  = z_std.view({1, 1, Z_DIM});
    torch::Tensor a_mean_b = a_mean.view({1, 1, A_DIM});
    torch::Tensor a_std_b  = a_std.view({1, 1, A_DIM});

    const std::string image_folder_path =
        "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/test_dir";

    std::vector<std::string> image_files;
    for (const auto &entry : fs::directory_iterator(image_folder_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
        {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
    cv::namedWindow("VAE Reconstruction", cv::WINDOW_NORMAL);
    cv::namedWindow("RNN Prediction", cv::WINDOW_NORMAL);

    std::deque<torch::Tensor> x_buf;
    uint32_t                  step = 0;

    torch::NoGradGuard no_grad;

    for (const auto &image_path : image_files)
    {
        const cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::imshow("Input Image", frame);

        const torch::Tensor vae_input = imageToTensor(frame, VAE_IMG_SIZE, device);

        const auto          encoder_output = vae_encoder.forward({vae_input}).toTuple();
        const torch::Tensor mu_t           = encoder_output->elements()[0].toTensor();
        const torch::Tensor logvar_t       = encoder_output->elements()[1].toTensor();

        // Get latent vector 'z' and reconstruct
        const torch::Tensor z_t     = reparameterize(mu_t, logvar_t);
        const torch::Tensor recon_t = vae_decoder.forward({z_t}).toTensor();

        cv::Mat vae_rec_img = tensorToImage(recon_t);
        cv::imshow("VAE Reconstruction", vae_rec_img);

        // 4. Run RNN prediction step
        if (step % STRIDE == 0)
        {
            float         acc = 100.0, other = 0.0;
            torch::Tensor a_t = torch::tensor({{acc, other}}, torch::kFloat32).view({1, 1, A_DIM}).to(device);

            torch::Tensor z_t_seq = z_t.view({1, 1, Z_DIM});
            torch::Tensor z_norm  = (z_t_seq - z_mean_b) / z_std_b;
            torch::Tensor a_norm  = (a_t - a_mean_b) / a_std_b;
            torch::Tensor x_t     = torch::cat({z_norm, a_norm}, -1);

            x_buf.push_back(x_t);
            if (x_buf.size() > SEQ_LEN)
            {
                x_buf.pop_front();
            }

            if (x_buf.size() == SEQ_LEN)
            {
                torch::Tensor x_seq      = torch::cat(std::vector<torch::Tensor>(x_buf.begin(), x_buf.end()), 1);
                auto          rnn_output = rnn_model.forward({x_seq}).toTuple();
                torch::Tensor z_next_pred_normalized = rnn_output->elements()[0].toTensor();

                torch::Tensor z_next = z_next_pred_normalized * z_std_b + z_mean_b;
                z_next               = z_next.view({1, Z_DIM});
                torch::Tensor pred   = vae_decoder.forward({z_next}).toTensor();

                cv::Mat pred_img = tensorToImage(pred);
                cv::imshow("RNN Prediction", pred_img);
            }
        }

        step++;
        if (cv::waitKey(30) == 27)
        {
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "Processing finished." << std::endl;
    return 0;
}