/* 
 * 3D Heat Conduction Equation Solver
 * 
 * This program discretizes the 3D heat conduction equation using an explicit first-order 
 * scheme in time, and the spatial derivatives are approximated using the second-order central difference method. 
 * 
 * The GPU kernel code is adapted from:
 * "Programming Massively Parallel Processors: A Hands-on Approach 4th Edition" by David B. Kirk and Wen-mei W. Hwu, 
 * Chapter 8.
 * 
 * Note: The code has been modified to suit the current application. 
 * Ensure to review the original source for detailed explanations.
 * 
 * Programmed by: Kazem Bazesefidpar
 * Email: bazesefidpar64@gmail.com
 * 
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include "common.cuh"
/*In the Makefile
 Kernel1: GPU solver with shared memory
 Kernel2: GPU solver with shared memory and thread coarsening in z-direction
 Kernel3: GPU solver with shared memory, thread coarsening in z-direction, and regiter tiling
*/
#define BX 8
#define BY 8
#define BZ 8   /*Input block size for shared memory kernel*/
#define BXSMTC 32
#define BYSMTC 32 /*Input block size for shared memory with thread coarsening kernel */

constexpr size_t NX = 480; constexpr size_t NY = 480; constexpr size_t NZ = 480;
constexpr float LX = 1.0f; constexpr float LY = 1.0f; constexpr float LZ = 1.0f;
constexpr float ALPHA = 0.00001f; constexpr float DT = 0.0000005f; constexpr float TIME = 0.001f;
constexpr float BCX = 0.0f; constexpr float TCX = 0.0f;
constexpr float BCY = 0.0f; constexpr float TCY = 0.0f;
constexpr float BCZ = 100.0f; constexpr float TCZ = 0.0f; /*BCZ : x-y plane on z=0; TCZ : x-y plane on z=1*/
constexpr size_t RADIUS{1}; constexpr size_t Dimension{7};

__constant__ float C3DGPU[Dimension];

/*GPU solver with just global memory*/
/*The kernel is adapted from 8.6 of the book*/
template<typename T>
__global__ void SolverHeatEquation3DWithoutSM(T *u, T *unew, const size_t nx, const size_t ny,
                                     const size_t nz) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {

    unew[nx * ny * k + ny * j + i] =
      C3DGPU[0]*u[nx*ny*k+ny*j+i] +
      C3DGPU[1]*u[nx*ny*k+ny*j+(i-1)] + C3DGPU[2]*u[nx*ny*k+ny*j+(i+1)] +
      C3DGPU[3]*u[nx*ny*k+ny*(j-1)+i] + C3DGPU[4]*u[nx*ny*k+ny*(j+1)+i] +
      C3DGPU[5]*u[nx*ny*(k-1)+ny*j+i] + C3DGPU[6]*u[nx*ny*(k+1)+ny*j+i];
  }
}
/*GPU solver using shared memory*/
/*The kernel is adapted from 8.8 of the book*/
template<typename T>
__global__ void SolverHeatEquation3DWithSM(T *u, T *unew, const size_t nx, const size_t ny,
                                   const size_t nz, const size_t radius) {

  int i = threadIdx.x + (blockDim.x-2*radius)*blockIdx.x-radius;
  int j = threadIdx.y + (blockDim.y-2*radius)*blockIdx.y-radius;
  int k = threadIdx.z + (blockDim.z-2*radius)*blockIdx.z-radius;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;

  __shared__ T un[BZ][BY][BX];

  if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
    un[z][y][x] = u[nx*ny*k+ny*j+i];
  }

  __syncthreads();

  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1 &&
      x >= radius && x < blockDim.x-radius &&
      y >= radius && y < blockDim.y-radius &&
      z >= radius && z < blockDim.z-radius) {

    unew[nx * ny * k + ny * j + i] =
      C3DGPU[0] * un[z][y][x] +
      C3DGPU[1] * un[z][y][x+1] + C3DGPU[2] * un[z][y][x-1] +
      C3DGPU[3] * un[z][y+1][x] + C3DGPU[4] * un[z][y-1][x] +
      C3DGPU[5] * un[z+1][y][x] + C3DGPU[6] * un[z-1][y][x];
  }
}
/*GPU solver using shared memory with thread coarsening*/
/*The kernel is adapted from 8.10 of the book*/
template<typename T>
__global__ void SolverHeatEquation3DWithSMThreadCoarsening(T *u, T *unew, const size_t nx,
		       const size_t ny, const size_t nz, const size_t radius) {

  size_t OUTTILEDIMZ{nz-2};
  int i = threadIdx.x + (blockDim.x-2*radius)*blockIdx.x-radius;
  int j = threadIdx.y + (blockDim.y-2*radius)*blockIdx.y-radius;
  int k{1};
  int x = threadIdx.x;
  int y = threadIdx.y;
  
  __shared__ T previous[BYSMTC][BXSMTC];
  __shared__ T current[BYSMTC][BXSMTC];
  __shared__ T next[BYSMTC][BXSMTC];
  
  if (i >= 0 && i < nx && j >= 0 && j < ny && (k-1) >= 0 && (k-1) < nz) {
    previous[y][x] = u[nx*ny*(k-1)+ny*j+i];
  }
  
  if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
    current[y][x] = u[nx*ny*k+ny*j+i];
  }
  
  for(int slice = k; slice< k+OUTTILEDIMZ ;++slice){
    if(i >= 0 && i < nx && j >= 0 && j < ny && (slice+1)>=0 && (slice+1)<nz){
      next[y][x] = u[nx*ny*(slice+1)+ny*j+i];
    }
    __syncthreads();
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && slice > 0 && slice < nz-1 &&
	x >= radius && x < blockDim.x-radius &&
	y >= radius && y < blockDim.y-radius) {
      
      unew[nx * ny * slice + ny * j + i] =
	C3DGPU[0] * current[y][x] +
	C3DGPU[1] * current[y][x+1] + C3DGPU[2] * current[y][x-1] +
	C3DGPU[3] * current[y+1][x] + C3DGPU[4] * current[y-1][x] +
	C3DGPU[5] * next[y][x]      + C3DGPU[6] * previous[y][x];
    }
    __syncthreads();
    previous[y][x] = current[y][x];
    current[y][x]  = next[y][x];
  }
} 
/*GPU solver using shared memory with thread coarsening and register tiling*/
/*The kernel is adapted from 8.12 of the book*/
template<typename T>
__global__ void SolverHeatEquation3DWithSMThreadCoarseningRegisterTiling(T *u, T *unew, const size_t nx,
                       const size_t ny, const size_t nz, const size_t radius) {
  
  size_t OUTTILEDIMZ{nz-2};
  int i = threadIdx.x + (blockDim.x-2*radius)*blockIdx.x-radius;
  int j = threadIdx.y + (blockDim.y-2*radius)*blockIdx.y-radius;
  int k{1};
  int x = threadIdx.x;
  int y = threadIdx.y;

  float previous, current, next;
  __shared__ T currentSM[BYSMTC][BXSMTC];
  
  if (i >= 0 && i < nx && j >= 0 && j < ny && (k-1) >= 0 && (k-1) < nz) {
    previous = u[nx*ny*(k-1)+ny*j+i];
  }
  if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
    current = u[nx*ny*k+ny*j+i];
    currentSM[y][x] = current;
  }
  for(int slice = k; slice< k+OUTTILEDIMZ ;++slice){
    if(i >= 0 && i < nx && j >= 0 && j < ny && (slice+1)>=0 && (slice+1)<nz){
      next = u[nx*ny*(slice+1)+ny*j+i];
    }
    __syncthreads();
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && slice > 0 && slice < nz-1 &&
        x >= radius && x < blockDim.x-radius &&
        y >= radius && y < blockDim.y-radius) {

      unew[nx * ny * slice + ny * j + i] =
        C3DGPU[0] * current +
        C3DGPU[1] * currentSM[y][x+1] + C3DGPU[2] * currentSM[y][x-1] +
	C3DGPU[3] * currentSM[y+1][x] + C3DGPU[4] * currentSM[y-1][x] +
        C3DGPU[5] * next              + C3DGPU[6] * previous;
    }
    __syncthreads();
    previous = current;
    current  = next;
    currentSM[y][x]  = next;
  }
} 

template<typename T>
__global__ void heatEquationUpdateSolution(T *u,T *unew,const size_t nx,const size_t ny,
					   const size_t nz){
  
  unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int j = threadIdx.y + blockDim.y*blockIdx.y;
  unsigned int k = threadIdx.z + blockDim.z*blockIdx.z;
  
  if(i<nx && j<ny && k<nz){
    u[nx*ny*k+ny*j+i] = unew[nx*ny*k+ny*j+i];
  }
}

int main(int argc,char* argv[]) {

  int device = 0;
  cudaSetDevice(device);
  
  /*Check the amount of the requested shared memory is adequate for the application*/
  size_t sharedMemMax{0};
  {
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp,device));
    sharedMemMax = iProp.sharedMemPerBlock;
  }
#ifdef Kernel1
  if(sharedMemMax<(BX*BY*BZ*sizeof(float))){
    std::cout<<"The amount of the shared memory per block "<<sharedMemMax<<
      ", but the requeted amount is "<<BX*BY*BZ*sizeof(float)<<std::endl;
    exit(EXIT_FAILURE);
  }
#elif Kernel2
  if(sharedMemMax<(BXSMTC*BYSMTC*sizeof(float))){
    std::cout<<"The amount of the shared memory per block is "<<sharedMemMax<<
      ", but the requeted amount is "<<BXSMTC*BYSMTC*sizeof(float)<<std::endl;
    exit(EXIT_FAILURE);
  }
#elif Kernel3
  if(sharedMemMax<(BXSMTC*BYSMTC*sizeof(float))){
    std::cout<<"The amount of the shared memory per block "<<sharedMemMax<<
      ", but the requeted amount is "<<BXSMTC*BYSMTC*sizeof(float)<<std::endl;
    exit(EXIT_FAILURE);
  }
#endif
  
  Coefficient<float> coe(ALPHA,LX,LY,LZ,NX,NY,NZ,DT,TIME,BCX,TCX,BCY,TCY,BCZ,TCZ);
  std::vector<float> C3D(Dimension,0.0);
  float epsError{0.001f};
  
  const size_t size{coe.nc[0]*coe.nc[1]*coe.nc[2]};
  std::vector<float> u(size,0.0);
#ifdef DEBUG
  std::vector<float> unew(size,0.0);
  Timer timerCPU("CPU");
#endif
  
  /*GPU allocation for kernel with using global memory*/
  std::vector<float> uGPUTest(size,0.0);
  float *uGPU{nullptr}, *unewGPU{nullptr};
  CHECK(cudaMalloc(reinterpret_cast<void**>(&uGPU),size*sizeof(float)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&unewGPU),size*sizeof(float)));
  Timer timer("without shared memory");
#ifdef Kernel1
  /*GPU allocation for kernel with using shared memory*/
  std::vector<float> uGPUTestSM(size,0.0);
  float *uGPUSM{nullptr}, *unewGPUSM{nullptr};
  CHECK(cudaMalloc(reinterpret_cast<void**>(&uGPUSM),size*sizeof(float)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&unewGPUSM),size*sizeof(float)));
  Timer timerSM("with shared memory");
#elif Kernel2
  /*GPU allocation for kernel with using shared memory and thread coarsening*/
  std::vector<float> uGPUTestSMTC(size,0.0);
  float *uGPUSMTC{nullptr}, *unewGPUSMTC{nullptr};
  CHECK(cudaMalloc(reinterpret_cast<void**>(&uGPUSMTC),size*sizeof(float)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&unewGPUSMTC),size*sizeof(float)));
  Timer timerSMTC("with shared memory and thread coarsening");
#elif Kernel3
    
  /*GPU allocation for kernel with using shared memory, thread coarsening, and register tiling */
  std::vector<float> uGPUTestSMTCRT(size,0.0);
  float *uGPUSMTCRT{nullptr}, *unewGPUSMTCRT{nullptr};
  CHECK(cudaMalloc(reinterpret_cast<void**>(&uGPUSMTCRT),size*sizeof(float)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&unewGPUSMTCRT),size*sizeof(float)));
  Timer timerSMTCRT("with shared memory, thread coarsening, and register tiling");
#endif
  /*Set the distretized 3D heat conduction equation coefficients*/
  setCoefficientEquation(C3D,coe);
  /*Copy C3D to C3DGPU allocated on GPU constant memory*/
  CHECK(cudaMemcpyToSymbol(C3DGPU,C3D.data(),Dimension*sizeof(float),0,cudaMemcpyHostToDevice));
  /*Set the boundary conditions on CPU*/
  apply_boundary_conditions(u,coe);
#ifdef DEBUG
  apply_boundary_conditions(unew,coe);
#endif
  /*Copy the initial field from CPU to GPU allocations*/
  CHECK(cudaMemcpy(uGPU,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(unewGPU,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
#ifdef Kernel1
  CHECK(cudaMemcpy(uGPUSM,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(unewGPUSM,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
#elif Kernel2
  CHECK(cudaMemcpy(uGPUSMTC,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(unewGPUSMTC,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
#elif Kernel3
  CHECK(cudaMemcpy(uGPUSMTCRT,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(unewGPUSMTCRT,u.data(),size*sizeof(float),cudaMemcpyHostToDevice));
#endif
  /*Set the block and grid sizes for different approach*/  
  dim3 block(BX-2*RADIUS,BY-2*RADIUS,BZ-2*RADIUS);
  dim3 grid((block.x+coe.nc[0]-1)/block.x,
            (block.y+coe.nc[1]-1)/block.y,
            (block.z+coe.nc[2]-1)/block.z);
#ifdef Kernel1
  dim3 blockSM(BX,BY,BZ);
#elif Kernel2
  dim3 blockSMTC(BXSMTC,BYSMTC);
  dim3 gridSMTC((blockSMTC.x-2*RADIUS+coe.nc[0]-1)/(blockSMTC.x-2*RADIUS),
		(blockSMTC.y-2*RADIUS+coe.nc[1]-1)/(blockSMTC.y-2*RADIUS));
#elif Kernel3
  dim3 blockSMTC(BXSMTC,BYSMTC);
  dim3 gridSMTC((blockSMTC.x-2*RADIUS+coe.nc[0]-1)/(blockSMTC.x-2*RADIUS),
                (blockSMTC.y-2*RADIUS+coe.nc[1]-1)/(blockSMTC.y-2*RADIUS));
#endif
  
  for (size_t n = 0; n < coe.nStep; ++n) {
#ifdef DEBUG
    /*Solve the discretized equations on the GPU*/
    timerCPU.start();
    for (size_t k = 1; k < coe.nc[2]-1; ++k){
      for (size_t j = 1; j < coe.nc[1]-1; ++j) {
	for (size_t i = 1; i < coe.nc[0]-1; ++i) {
	  unew[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*j+i] =
	    C3D[0]*u[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*j+i]+
	    C3D[1]*u[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*j+(i-1)]+
	    C3D[2]*u[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*j+(i+1)]+
	    C3D[3]*u[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*(j-1)+i]+
	    C3D[4]*u[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*(j+1)+i]+
	    C3D[5]*u[coe.nc[0]*coe.nc[1]*(k-1)+coe.nc[1]*j+i]+
            C3D[6]*u[coe.nc[0]*coe.nc[1]*(k+1)+coe.nc[1]*j+i];
	}
      }
    }
    timerCPU.stop(false);
    std::swap(u,unew);
#endif
    
    /*GPU solver with just global memory*/
    timer.start();
    SolverHeatEquation3DWithoutSM<<<grid,block>>>(uGPU,unewGPU,coe.nc[0],coe.nc[1],coe.nc[2]);
    CHECK(cudaDeviceSynchronize());
    timer.stop(false);
    CHECK(cudaGetLastError());

#ifdef Kernel1
    /*GPU solver using shared memory*/
    timerSM.start();
    SolverHeatEquation3DWithSM<<<grid,blockSM>>>(uGPUSM,unewGPUSM,coe.nc[0],coe.nc[1],coe.nc[2],RADIUS);
    CHECK(cudaDeviceSynchronize());
    timerSM.stop(false);
    CHECK(cudaGetLastError());
    
#elif Kernel2
    /*GPU solver using shared memory with thread coarsening*/    
    timerSMTC.start();
    SolverHeatEquation3DWithSMThreadCoarsening<<<gridSMTC,blockSMTC>>>(uGPUSMTC,unewGPUSMTC,coe.nc[0],coe.nc[1],coe.nc[2],RADIUS);
    CHECK(cudaDeviceSynchronize());
    timerSMTC.stop(false);
    CHECK(cudaGetLastError());
#elif Kernel3
    /*GPU solver using shared memory with thread coarsening and register tiling*/
    timerSMTCRT.start();
    SolverHeatEquation3DWithSMThreadCoarseningRegisterTiling<<<gridSMTC,blockSMTC>>>(uGPUSMTCRT,unewGPUSMTCRT,
										     coe.nc[0],coe.nc[1],coe.nc[2],RADIUS);
    CHECK(cudaDeviceSynchronize());
    timerSMTCRT.stop(false);
    CHECK(cudaGetLastError());
#endif
    /*Put the T{n} = T{n+1} for the next iteration
     all kernels are launched on the same grid*/
    heatEquationUpdateSolution<<<grid,block>>>(uGPU,unewGPU,coe.nc[0],coe.nc[1],coe.nc[2]);
#ifdef Kernel1
    heatEquationUpdateSolution<<<grid,block>>>(uGPUSM,unewGPUSM,coe.nc[0],coe.nc[1],coe.nc[2]);
#elif Kernel2    
    heatEquationUpdateSolution<<<grid,block>>>(uGPUSMTC,unewGPUSMTC,coe.nc[0],coe.nc[1],coe.nc[2]);
#elif Kernel3
    heatEquationUpdateSolution<<<grid,block>>>(uGPUSMTCRT,unewGPUSMTCRT,coe.nc[0],coe.nc[1],coe.nc[2]);
#endif
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    /*Test the result for correctness*/
    if (n % 400 == 0){
      std::cout<<"Test at n = "<<n<<std::endl;
      cudaMemcpy(uGPUTest.data(),unewGPU,size*sizeof(float),cudaMemcpyDeviceToHost);
#ifdef Kernel1
      cudaMemcpy(uGPUTestSM.data(),unewGPUSM,size*sizeof(float),cudaMemcpyDeviceToHost);
      for(size_t i=0;i<uGPUTestSM.size();++i){
        if(std::fabs(uGPUTest[i]-uGPUTestSM[i])>epsError){
          std::cerr<<"Shared memory : Error["<<i<<"] = "<<uGPUTest[i]-uGPUTestSM[i]<<"\n";
          break;
        }
      }
#elif Kernel2
      cudaMemcpy(uGPUTestSMTC.data(),unewGPUSMTC,size*sizeof(float),cudaMemcpyDeviceToHost);
      for(size_t i=0;i<uGPUTestSMTC.size();++i){
        if(std::fabs(uGPUTest[i]-uGPUTestSMTC[i])>epsError){
          std::cerr<<"Shared memory with TC : Error["<<i<<"] = "<<uGPUTest[i]-uGPUTestSMTC[i]<<"\n";
          break;
        }
      }
#elif Kernel3
      cudaMemcpy(uGPUTestSMTCRT.data(),unewGPUSMTCRT,size*sizeof(float),cudaMemcpyDeviceToHost);
      for(size_t i=0;i<uGPUTestSMTCRT.size();++i){
        if(std::fabs(uGPUTest[i]-uGPUTestSMTCRT[i])>epsError){
          std::cerr<<"Shared memory with TC and RT : Error["<<i<<"] = "<<uGPUTest[i]-uGPUTestSMTCRT[i]<<"\n";
          break;
        }
      }
#endif

#ifdef DEBUG
      for(size_t i=0;i<uGPUTest.size();++i){
	if(std::abs(uGPUTest[i]-unew[i])>0.0001){
	  std::cerr<<"Error["<<i<<"] = "<<uGPUTest[i]-unew[i]<<"\n";
	  break;
	}
      }
#endif
    }
  }
#ifdef DEBUG
  timerCPU.averageTime();
#endif
  /*Print the average computation time for different kernels*/
  timer.averageTime();
#ifdef Kernel1
  timerSM.averageTime();
#elif Kernel2
  timerSMTC.averageTime(); 
#elif Kernel3
  timerSMTCRT.averageTime();
#endif
  
  std::cout << "Simulation complete" << std::endl;
  /*Free the resources*/
  CHECK(cudaFree(uGPU));
  CHECK(cudaFree(unewGPU));
#ifdef Kernel1
  CHECK(cudaFree(uGPUSM));
  CHECK(cudaFree(unewGPUSM));
#elif Kernel2
  CHECK(cudaFree(uGPUSMTC));
  CHECK(cudaFree(unewGPUSMTC));
#elif Kernel3
  CHECK(cudaFree(uGPUSMTCRT));
  CHECK(cudaFree(unewGPUSMTCRT));
#endif
  return EXIT_SUCCESS;
}
