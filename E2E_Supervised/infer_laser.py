import math
import struct
from multiprocessing import shared_memory

import posix_ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import TransformerModel

shm = shared_memory.SharedMemory(name="myshm")
sem1 = posix_ipc.Semaphore("/sem1", posix_ipc.O_CREAT, initial_value=0)
sem2 = posix_ipc.Semaphore("/sem2", posix_ipc.O_CREAT, initial_value=0)

model = TransformerModel()

model.load_state_dict(torch.load("simple_nn.pth"))

model.eval()

with torch.no_grad():
    while True:  # Continuous loop
        # Wait for C++ to signal that data is ready
        sem1.acquire()

        # Read data from shared memory
        data = shm.buf[:56]  # Adjust size based on your data structure
        serialized_points = struct.unpack("f" * 14, data)
        measurement_points = [
            (serialized_points[i], serialized_points[i + 1])
            for i in range(0, len(serialized_points), 2)
        ]
        print(f"Received points: {measurement_points}")

        # Create tensor from the received data
        points_tensor = torch.tensor(measurement_points).view(1, 7, 2)

        # Infer the model
        action = model(points_tensor)

        # Send the response
        response = tuple(action.numpy().flatten())
        print(response)
        shm.buf[56:64] = struct.pack("ff", *response)
        # Signal C++ that response is ready
        sem2.release()


example_input = torch.rand((1, 7, 2))  # Shape: (batch_size, sequence_length, input_dim)
