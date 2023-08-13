#include "Msg.h"
#include "spmc_queue.h"

using namespace std;

// use taskset -c to bind core
int main(int argc, char **argv)
{
    const char *shm_file = "SPMCQueue_test";

    Q<Msg, 4> *q = shmmap<Msg, 4>(shm_file);
    if (!q)
        return 1;

    uint32_t i = 0;
    while (true)
    {
        q->write(
            [i](Msg &msg)
            {
                msg.idx      = i;
                msg.data[10] = i * 7;
            });
        i++;
        std::cout << "wrote " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
