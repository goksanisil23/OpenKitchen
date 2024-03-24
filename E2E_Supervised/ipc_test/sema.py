import struct
import time
from multiprocessing import shared_memory

import posix_ipc

shm = shared_memory.SharedMemory(name="myshm")
sem1 = posix_ipc.Semaphore("/sem1", posix_ipc.O_CREAT, initial_value=0)
sem2 = posix_ipc.Semaphore("/sem2", posix_ipc.O_CREAT, initial_value=0)

ctr = 0
while True:  # Continuous loop
    # Wait for C++ to signal that data is ready
    sem1.acquire()

    # Read data from shared memory
    data = shm.buf[:56]  # 56 = 7*2 points * 4 bytes
    serialized_points = struct.unpack("f" * 14, data)
    points = [
        (serialized_points[i], serialized_points[i + 1])
        for i in range(0, len(serialized_points), 2)
    ]
    print(f"Received points: {points}")

    # Action as response = 2 actions * 4 bytes
    response = (4.56, ctr)  # Dummy response
    shm.buf[56:64] = struct.pack("ff", *response)

    # Signal C++ that response is ready
    sem2.release()

    ctr += 1
