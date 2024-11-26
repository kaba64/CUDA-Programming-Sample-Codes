# Computing Histogram using CUDA

This educational code demonstrates techniques to compute the parallel scan of an arbitrary-length inputs vector, focusing on CUDA-based optimizations, particularly through shuffle function and using C++ memory model for the synchronization.

The optimization techniques in this code are adapted from the following references:

- **[1]** *Programming Massively Parallel Processors: A Hands-on Approach (4th Edition)* by David B. Kirk and Wen-mei W. Hwu
- **[2]** *https://developer.nvidia.com/blog/faster-parallel-reductions-kepler
- **[3]** *https://en.cppreference.com/w/cpp/atomic/memory_order
- **[4]** *https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html

## GPU Kernel Configurations

The code computes the parallel scan of input vector by using shuffle function and StreamScan and Decoupled Look-back algorithms.