This educational code is designed to help become familiar with key GPU optimization techniques, specifically thread coarsening and register tiling, by solving the 3D heat conduction equation using the CUDA programming model. The optimizations and techniques used in this code are adapted from Chapter 8 of the book Programming Massively Parallel Processors: A Hands-on Approach (4th Edition) by David B. Kirk and Wen-mei W. Hwu, which serves as a key reference for understanding GPU programming concepts.

The heat conduction equation being solved is:

This is a sample code for solving the 3D heat conduction equation on a single GPU using the CUDA programming model:

$$
\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2} \right)
$$

where \( $$\alpha \$$) is the thermal diffusivity and \( T \) is the temperature.

### Time Discretization

The explicit first-order scheme in time (Forward Euler method) is used:

$$
\frac{T^{n+1} - T^n}{\Delta t} = \alpha \left( \frac{\partial^2 T^n}{\partial x^2} + \frac{\partial^2 T^n}{\partial y^2} + \frac{\partial^2 T^n}{\partial z^2} \right)
$$

Rearranging this for \( $$T^{n+1} $$\), we get:

$$
T^{n+1} = T^n + \alpha \Delta t \left( \frac{\partial^2 T^n}{\partial x^2} + \frac{\partial^2 T^n}{\partial y^2} + \frac{\partial^2 T^n}{\partial z^2} \right)
$$

### Spatial Discretization

The spatial derivatives \( $$\frac{\partial^2 T}{\partial x^2}$$ \), \( $$\frac{\partial^2 T}{\partial y^2} $$\), and \( $$\frac{\partial^2 T}{\partial z^2} $$\) are approximated using the second-order central difference method:

- In the \(x\)-direction:

$$
\frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1,j,k} - 2T_{i,j,k} + T_{i-1,j,k}}{(\Delta x)^2}
$$

- Similarly for the \(y\) and \(z\)-directions.

### Final Discretized Equation

The final update equation for $$T^{n+1}_{i,j,k} $$ at any grid point $$(i,j,k) $$ becomes:

$$
\begin{align*}
T^{n+1}\_{i,j,k} &= T^n\_{i,j,k} + \alpha \Delta t \left( \frac{T^n\_{i+1,j,k} - 2T^n\_{i,j,k} + T^n\_{i-1,j,k}}{(\Delta x)^2} \right. \\
&\quad + \frac{T^n\_{i,j+1,k} - 2T^n\_{i,j,k} + T^n\_{i,j-1,k}}{(\Delta y)^2} \\
&\quad \left. + \frac{T^n\_{i,j,k+1} - 2T^n\_{i,j,k} + T^n\_{i,j,k-1}}{(\Delta z)^2} \right)
\end{align*}
$$

### GPU kernel codes

The discretized equation is solved on CPU and GPU using CUDA programig model. Four different kernels (GPU functions) have been implemented to improve the efficiency of the computation on the GPU. Their computation time and accuracy have been compared. The final version of the kernel employs shared memory, thread coarsening, and register tiling for optimization.

For a detail explanation of the methds used for the GPU kernels (the GPU kernel codes are adapted from this book), please see the chapter 8 of the book:

Programming Massively Parallel Processors: A Hands-on Approach 4th Edition" by David B. Kirk and Wen-mei W. Hwu


### Computationl Geometry
 To apply thread coarsening, coarsening across the entire grid in the z-dimension is used. Please see the provided figure for thread coarsening in the z-direction. 
 
![3D cube with pencils in z-direction](ch2/cube.png)

### Makefile

In the Makefile:

DEBUG  : CPU solver
Kernel1: GPU solver with shared memory                                                                                              
Kernel2: GPU solver with shared memory and thread coarsening in z-direction                                                         
Kernel3: GPU solver with shared memory, thread coarsening in z-direction, and regiter tiling
