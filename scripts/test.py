#!/usr/bin/env python3
"""
benchmark_neural_network.py - pytorch neural network benchmark
mirrors the STM32 implementation with explicit tensor operations
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MNISTNetwork(nn.Module):
    """
    simple dense network matching STM32 architecture
    784 input -> 128 hidden -> 64 hidden -> 10 output
    """
    
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        
        # define linear layers - these hold weight and bias tensors
        self.fc1 = nn.Linear(784, 128)  # weight shape: (128, 784), bias shape: (128,)
        self.fc2 = nn.Linear(128, 64)   # weight shape: (64, 128), bias shape: (64,)  
        self.fc3 = nn.Linear(64, 10)    # weight shape: (10, 64), bias shape: (10,)
        
        # initialize weights using xavier normal - prevents vanishing/exploding gradients
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight) 
        nn.init.xavier_normal_(self.fc3.weight)
        
        # initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        """
        forward pass - explicitly show each tensor operation
        input x shape: (batch_size, 784)
        """
        # first layer: linear transformation followed by relu
        # x @ fc1.weight.T + fc1.bias -> (batch_size, 784) @ (784, 128) + (128,) = (batch_size, 128)
        h1 = self.fc1(x)           # linear: y = xW^T + b
        h1_relu = F.relu(h1)       # relu: max(0, h1) element-wise
        
        # second layer: (batch_size, 128) -> (batch_size, 64)
        h2 = self.fc2(h1_relu)     # linear transformation
        h2_relu = F.relu(h2)       # relu activation
        
        # output layer: (batch_size, 64) -> (batch_size, 10)
        output = self.fc3(h2_relu) # final linear layer - raw logits
        
        return output              # return raw scores, not softmax probabilities

def create_test_input():
    """
    create exact same input pattern as STM32 version
    first 10 elements = 0.1, remaining = 0.0
    """
    # create tensor filled with zeros - dtype=float32 for consistency with STM32
    input_tensor = torch.zeros(1, 784, dtype=torch.float32)
    
    # set first 10 elements to 0.1 - matches STM32 input exactly
    input_tensor[0, :10] = 0.1
    
    return input_tensor

def load_or_create_model():
    """
    attempt to load pre-trained model, create new one if not found
    """
    model = MNISTNetwork()
    
    try:
        # try loading saved state dict
        state_dict = torch.load('mnist_model.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        print("loaded existing model from mnist_model.pth")
    except FileNotFoundError:
        print("no saved model found - using randomly initialized weights")
        print("note: predictions will be random without training")
        
        # save the randomly initialized model for consistency
        torch.save(model.state_dict(), 'mnist_model.pth')
        print("saved initialized model to mnist_model.pth")
    
    # set to evaluation mode - disables dropout, batch norm updates
    model.eval()
    
    return model

def benchmark_inference(model, input_data, iterations=10):
    """
    run inference timing benchmark with torch.no_grad() context
    """
    print(f"running {iterations} inference iterations...")
    
    iteration_times = []
    
    # torch.no_grad() disables gradient computation for faster inference
    with torch.no_grad():
        # warm up - first inference often slower due to memory allocation
        _ = model(input_data)
        
        for iteration in range(iterations):
            print(f"iteration {iteration + 1}: ", end="")
            
            # high precision timing using perf_counter
            start_time = time.perf_counter()
            
            # forward pass through network
            logits = model(input_data)
            
            # end timing immediately after forward pass
            end_time = time.perf_counter()
            
            # calculate elapsed time in seconds
            elapsed_time = end_time - start_time
            iteration_times.append(elapsed_time)
            
            # get prediction - argmax of logits gives predicted class
            # logits shape: (1, 10), we want index of maximum value
            predicted_digit = torch.argmax(logits, dim=1).item()
            
            # convert timing to milliseconds and microseconds
            elapsed_ms = elapsed_time * 1000.0
            elapsed_us = elapsed_time * 1000000.0
            
            print(f"{elapsed_ms:.3f} ms ({elapsed_us:.1f} us), prediction: {predicted_digit}")
    
    # calculate statistics
    average_time = sum(iteration_times) / len(iteration_times)
    inferences_per_second = 1.0 / average_time
    
    return iteration_times, average_time, inferences_per_second

def display_benchmark_results(iteration_times, average_time, inferences_per_second):
    """
    display benchmark statistics matching STM32 output format
    """
    print("\n=== BENCHMARK RESULTS ===")
    
    total_time = sum(iteration_times)
    print(f"total time: {total_time:.6f} seconds")
    
    avg_ms = average_time * 1000.0
    avg_us = average_time * 1000000.0
    print(f"average time: {avg_ms:.3f} ms ({avg_us:.1f} us)")
    print(f"inference rate: {inferences_per_second:.2f} inferences/sec")
    
    # show timing distribution
    min_time = min(iteration_times) * 1000.0
    max_time = max(iteration_times) * 1000.0
    print(f"timing range: {min_time:.3f} ms to {max_time:.3f} ms")

def compare_with_stm32_results(avg_time_ms):
    """
    compare pytorch performance with STM32 benchmark
    """
    # your STM32 benchmark results
    stm32_time_ms = 155.498
    stm32_rate = 6.43
    
    python_rate = 1000.0 / avg_time_ms
    
    print(f"\n=== PERFORMANCE COMPARISON ===")
    print(f"STM32 F446RE:      {stm32_time_ms:.3f} ms per inference ({stm32_rate:.2f} inf/sec)")
    print(f"PyTorch (this PC): {avg_time_ms:.3f} ms per inference ({python_rate:.2f} inf/sec)")
    
    if avg_time_ms < stm32_time_ms:
        speedup = stm32_time_ms / avg_time_ms
        print(f"PyTorch is {speedup:.1f}x faster than STM32")
    else:
        slowdown = avg_time_ms / stm32_time_ms
        print(f"STM32 is {slowdown:.1f}x faster than PyTorch")

def analyze_model_weights(model):
    """
    examine the model's internal structure and weight distributions
    """
    print("\n=== MODEL ANALYSIS ===")
    
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()  # number of elements in tensor
        total_params += num_params
        
        print(f"{name}: shape {list(param.shape)}, {num_params} parameters")
        print(f"  weight range: [{param.min().item():.4f}, {param.max().item():.4f}]")
        print(f"  weight std: {param.std().item():.4f}")
    
    print(f"total parameters: {total_params}")
    
    # calculate memory usage - float32 = 4 bytes per parameter
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"model memory: {memory_mb:.2f} MB")

def main():
    """
    main execution function
    """
    print("🔧 pytorch neural network benchmark")
    print("===================================")
    
    # check if cuda is available for gpu acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # create and load model
    model = load_or_create_model()
    model = model.to(device)  # move model to gpu if available
    
    # analyze model structure and weights
    analyze_model_weights(model)
    
    # create test input matching STM32 implementation
    print("\npreparing input data...")
    input_data = create_test_input()
    input_data = input_data.to(device)  # move input to same device as model
    
    print(f"input tensor shape: {input_data.shape}")
    print(f"input tensor dtype: {input_data.dtype}")
    print(f"first 10 elements: {input_data[0, :10].tolist()}")
    print(f"elements 10-20: {input_data[0, 10:20].tolist()}")
    
    # run benchmark with same iteration count as STM32
    iteration_times, average_time, inferences_per_second = benchmark_inference(
        model, input_data, iterations=10
    )
    
    # display results
    display_benchmark_results(iteration_times, average_time, inferences_per_second)
    
    # compare with STM32 performance
    avg_time_ms = average_time * 1000.0
    compare_with_stm32_results(avg_time_ms)

if __name__ == "__main__":
    main()