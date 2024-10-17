This educational code is shared to help users become familiar with parallel reduction techniques such as sequential addressing, shared memory usage, atomic operations, and warp-level primitives.

The optimization techniques used in this code are adapted from the following references:

[1] Harris, M., 2007. Optimizing Parallel Reduction in CUDA. Nvidia Developer Technology, 2(4), p.70.

[2] https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

[3] Programming Massively Parallel Processors: A Hands-on Approach (4th Edition) by David B. Kirk and Wen-mei W. Hwu


### GPU Kernel Codes

The code computes the length of a vector of size N using random numbers.

The length of the vector is computed on the CPU using parallel algorithms from std::execution, and also with different GPU kernels using the CUDA programming model.


### Makefile

The code requires the Intel Threading Building Blocks (TBB) library (-ltbb). Please ensure the correct directory path for TBB is specified according to your system's configuration.
