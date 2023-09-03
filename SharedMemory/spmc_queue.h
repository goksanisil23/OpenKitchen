#pragma once

#include <atomic>
#include <fcntl.h>
#include <iostream>
#include <string.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>

template <class T, uint32_t CNT>
class SPMCQueue
{
  public:
    static_assert(CNT && !(CNT & (CNT - 1)), "CNT must be a power of 2");
    struct Reader
    {
        operator bool() const
        {
            return q;
        }
        T *read()
        {
            auto    &blk     = q->blks[next_idx % CNT];
            uint32_t new_idx = ((std::atomic<uint32_t> *)&blk.idx)->load(std::memory_order_acquire);
            if (int(new_idx - next_idx) < 0)
                return nullptr;
            next_idx = new_idx + 1;
            return &blk.data;
        }

        T *readLast()
        {
            T *ret = nullptr;
            while (T *cur = read())
                ret = cur;
            return ret;
        }

        SPMCQueue<T, CNT> *q = nullptr;
        uint32_t           next_idx;
    };

    Reader getReader()
    {
        Reader reader;
        reader.q        = this;
        reader.next_idx = write_idx + 1;
        return reader;
    }

    template <typename Writer>
    void write(Writer writer)
    {
        auto &blk = blks[++write_idx % CNT];
        writer(blk.data);
        ((std::atomic<uint32_t> *)&blk.idx)->store(write_idx, std::memory_order_release);
    }

  private:
    friend class Reader;
    struct alignas(64) Block
    {
        uint32_t idx = 0;
        T        data;
    } blks[CNT];

    alignas(128) uint32_t write_idx = 0;
};

template <typename TUserMsg, size_t QueueSize>
using Q = SPMCQueue<TUserMsg, QueueSize>;

template <typename TUserMsg, size_t QueueSize>
inline Q<TUserMsg, QueueSize> *shmmap(const char *filename)
{
    int fd = shm_open(filename, O_CREAT | O_RDWR, 0666);
    if (fd == -1)
    {
        std::cerr << "shm_open failed: " << strerror(errno) << std::endl;
        return nullptr;
    }
    if (ftruncate(fd, sizeof(Q<TUserMsg, QueueSize>)))
    {
        std::cerr << "ftruncate failed: " << strerror(errno) << std::endl;
        close(fd);
        return nullptr;
    }
    Q<TUserMsg, QueueSize> *ret =
        (Q<TUserMsg, QueueSize> *)mmap(0, sizeof(Q<TUserMsg, QueueSize>), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ret == MAP_FAILED)
    {
        std::cerr << "mmap failed: " << strerror(errno) << std::endl;
        return nullptr;
    }
    return ret;
}
