#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main()
{
    int shm_fd = shm_open("myshm", O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, 1024);
    void *ptr = mmap(0, 1024, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);

    sem_t *sem1 = sem_open("/sem1", O_CREAT, 0666, 0);
    sem_t *sem2 = sem_open("/sem2", O_CREAT, 0666, 0);

    float points[14] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    while (true)
    {
        points[0]++;
        memcpy(ptr, points, sizeof(points));

        // Signal Python that data is ready
        sem_post(sem1);

        // Wait for Python to process and respond
        sem_wait(sem2);

        // Read response
        float response[2];
        memcpy(response, static_cast<char *>(ptr) + sizeof(points), sizeof(response));
        std::cout << "Response received: " << response[0] << ", " << response[1] << std::endl;
    }
}
