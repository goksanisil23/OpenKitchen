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

    auto reader = q->getReader();
    cout << "reader size: " << sizeof(reader) << endl;

    while (true)
    {
        Msg *msg = reader.readLast();
        if (!msg)
        {
            std::cout << "No msg yet..." << std::endl;
            continue;
        }
        cout << "i: " << msg->idx << std::endl;
        cout << "data: " << msg->data[10] << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}
