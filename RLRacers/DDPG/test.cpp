#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/torch.h>
#include <vector>

struct SubModule : torch::nn::Module
{
    SubModule()
    {
        // Register a parameter in the submodule
        weight = register_parameter("weight", torch::randn({10}));
    }
    torch::Tensor weight;
};

struct MainModule : torch::nn::Module
{
    MainModule()
    {
        // Register a submodule and a parameter in the main module
        submodule = register_module("submodule", std::make_shared<SubModule>());
        bias      = register_parameter("bias", torch::randn({10}));
    }
    std::shared_ptr<SubModule> submodule;
    torch::Tensor              bias;
};

int main()
{
    MainModule mainModule;

    // With recurse=true, it includes parameters from both mainModule and its submodule
    for (const auto &param : mainModule.named_parameters(true))
    {
        std::cout << "Param name: " << param.key() << std::endl;
    }
    std::cout << "------------" << std::endl;

    // With recurse=false, it includes parameters from only the mainModule
    for (const auto &param : mainModule.named_parameters(false))
    {
        std::cout << "Param name: " << param.key() << std::endl;
    }

    return 0;
}
