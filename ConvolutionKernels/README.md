# 2D Convolution with Gaussian Blur Filter using CUDA

This educational code demonstrates 2D convolution techniques to help users understand CUDA-based optimization, particularly using shared and constant memory. The code applies a Gaussian blur filter on an image using different GPU kernel configurations for efficient processing.

The optimization techniques in this code are adapted from the following reference:

- **[1]** *Programming Massively Parallel Processors: A Hands-on Approach (4th Edition)* by David B. Kirk and Wen-mei W. Hwu

## GPU Kernel Configurations

The code performs a 2D convolution on the image’s color channels (BGR) using a Gaussian blur filter. Two shared memory configurations are used to demonstrate different optimization approaches:

1. **Configuration 1**: Shared memory size matches the thread block size, offering simplicity.

   ![Configuration 1](image/fig_1.png)

2. **Configuration 2**: Shared memory size is larger than the thread block, enabling better handling of boundary conditions.

   ![Configuration 2](image/fig_3.png)

## Input and Output Example

Below is an example of the Gaussian blur filter applied to an image:

<div style="display: flex; justify-content: space-around;">
    <img src="src/sample.jpg" alt="Original Image" width="45%"/>
    <img src="src/convolved_image.png" alt="Blurred Image" width="45%"/>
</div>

- **Left**: The original image.
- **Right**: The image after applying the Gaussian blur filter.
