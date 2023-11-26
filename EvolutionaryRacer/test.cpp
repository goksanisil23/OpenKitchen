

#include <iostream>
#include <random>
#include <vector>

int main()
{
    std::mt19937                     gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < 10; i++)
    {
        double random_number = dis(gen);
        std::cout << random_number << '\n';
    }
    return 0;
}
