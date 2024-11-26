/*
 * Computation of parallel scan for arbitrary-length inputs 
 *
 * This program computes the inclusive Scan of a vector with size N using shuffle instruction and StreamScan algorithm.  
 *
 * Some of GPU kernel codes are adapted from the following references:
 *
 * [1] "Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition" by David B. Kirk and Wen-mei W. Hwu
 *
 * [2] https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 *
 * [3] https://en.cppreference.com/w/cpp/atomic/memory_order
 *
 * [4] https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html
 *
 * Note: The code has been modified to suit this application.                                                                 
 * Please review the original sources for detailed explanations.                                                                   
 *
 * Programmed by: Kazem Bazesefidpar
 * Email: bazesefidpar64@gmail.com
 *
 */
#include <iostream>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cuda/std/atomic>
#include <cuda_runtime.h>
#include "./../../common.cuh"


constexpr size_t N{1<<22};
constexpr size_t Nx{1024};
constexpr size_t gridDimX = (N+Nx-1)/Nx;
using VarType = long long int;

template<typename T>
void sequentialInclusiveScan (const T *x, T *y, const size_t size){
  y[0] = x[0];
  for(size_t i=1;i<size;++i)
    y[i]=y[i-1]+x[i];
}

template<typename T>
__inline__ __device__ T inclusiveScanShuffleBlock(const unsigned int & i, T *warpSM, const T *x, const unsigned int n){
  /*Convert thread block variables to the warp level variables*/
  unsigned int warpId    = threadIdx.x/warpSize;
  unsigned int warpIdMax = (blockDim.x+warpSize-1)/warpSize;
  unsigned int warpIndex = threadIdx.x%warpSize;
  unsigned mask          = __ballot_sync(0xffffffff,i<n);

  /*Apply Kogge-Stone algorithm within each warp*/
  T scanInWarp = (i<n) ? x[i] : static_cast<T>(0);
  #pragma unroll
  for(unsigned int stride=1;stride<warpSize;stride*=2){
    T var = __shfl_up_sync(mask,scanInWarp,stride);
    if(warpIndex>=stride)
      scanInWarp+=var;
  }
   /* Store the result from the last thread in each warp to shared memory */
  if (((warpIndex == warpSize - 1) && (i<n)) || ((i == n-1) && ((i+1)%warpSize) != 0))
    warpSM[warpId] = scanInWarp;
  
  __syncthreads();
  /* Use the first warp (warp 0) to perform an scan on the warp results in shared memory */
  if(threadIdx.x<warpSize){
    T val = threadIdx.x<warpIdMax ? warpSM[threadIdx.x]:static_cast<T>(0);
    mask  = __ballot_sync(0xffffffff,threadIdx.x<warpIdMax);
    #pragma unroll
    for(unsigned int stride=1;stride<warpIdMax;stride*=2){
      T var = __shfl_up_sync(mask,val,stride);
      if(threadIdx.x>=stride)
        val+=var;
    }
    if(threadIdx.x<warpIdMax)
      warpSM[threadIdx.x] = val; // Update shared memory results with prefix sums*/
  }
  __syncthreads();
  return (warpId!=0) ? scanInWarp+warpSM[warpId-1] : scanInWarp;

}
/*It is adopted from references [1].*/
template<typename T>
__global__ void inclusiveScanShuffleGPU(const T *x,T *y, const unsigned int n,
					int *flag, T *scanBlockShared, int *counter){
  extern __shared__ T warpSM[];
  __shared__ int blockIdxDynamicSh;
  __shared__ T previousBlockLastElementSum;
  /*Compute the thread blocks' Id using dynamic block indexing*/
  if(threadIdx.x==blockDim.x-1){
    blockIdxDynamicSh = atomicAdd(counter,1);
  }
  __syncthreads();
  /*Calculate global thread index*/
  const int blockIdxDynamic = blockIdxDynamicSh;
  unsigned int i  = blockDim.x*blockIdxDynamic+threadIdx.x;
  /*Perform inclusive scan per thread block using shuffle function*/
  T scanInBlock = inclusiveScanShuffleBlock(i,warpSM,x,n);
  /*Synchronization and partial sum propagation*/
  if(((threadIdx.x==blockDim.x-1) && (i<n)) || ((i==n-1) && ((i+1)%blockDim.x!=0))){
    if(blockIdxDynamic!=0){
      cuda::atomic_ref<int, cuda::thread_scope_device> fFlagI(flag[blockIdxDynamic]);
      while(fFlagI.load(cuda::memory_order_acquire)==0);
      previousBlockLastElementSum          = scanBlockShared[blockIdxDynamic];
      if(blockIdxDynamic<gridDim.x-1)
	scanBlockShared[blockIdxDynamic+1] = previousBlockLastElementSum+scanInBlock;
    }else{
      scanBlockShared[blockIdxDynamic+1] = scanInBlock;
    }
    //__threadfence(); /*No need if your CUDA Toolkit (>10.2) supports libcu++ */
    if(blockIdxDynamic<gridDim.x-1){
      cuda::atomic_ref<int, cuda::thread_scope_device> fFlagIP(flag[blockIdxDynamic+1]);
      fFlagIP.store(1,cuda::memory_order_release);
    }
  }
  __syncthreads();
  /*Write the final result to the output array*/
  if(i<n){
    y[i] = blockIdxDynamic==0 ? scanInBlock : scanInBlock+previousBlockLastElementSum;
  }
}      
/*
 * Main function to execute the parallel inclusive scan
 * - Initializes input data
 * - Allocates GPU memory
 * - Performs CPU and GPU inclusive scans
 * - Compares results for correctness
 */
int main(int argc, char* argv[]) {

  int device{0};
  cudaDeviceProp iProp;
  CHECK(cudaSetDevice(device));
  CHECK(cudaGetDeviceProperties(&iProp,device));
  
  size_t sharedMemoryWarp{sizeof(VarType)*((::Nx+iProp.warpSize-1)/iProp.warpSize)};
  {
    if(sharedMemoryWarp>iProp.sharedMemPerBlock){
      std::cout<<"The amount of the shared memory per block "<<iProp.sharedMemPerBlock
               <<", but the requeted amount for Warp is "<<sharedMemoryWarp<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  std::unique_ptr<VarType[]> XCPU        = std::make_unique<VarType[]>(::N);
  std::unique_ptr<VarType[]> SSCPU       = std::make_unique<VarType[]>(::N);
  std::unique_ptr<VarType[]> SSSHGPUTest = std::make_unique<VarType[]>(::N);
  
  //size_t initialData{0};

  /*Allocate input data on GPU*/
  VarType *XGPU{nullptr};
  /*Allocate output data on GPU*/
  VarType *SSSHGPU{nullptr}, *scanBlockShared{nullptr};
  int *flag{nullptr}, *counter{nullptr};

  CHECK(cudaMalloc(reinterpret_cast<void **>(&XGPU),sizeof(VarType)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&SSSHGPU),sizeof(VarType)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&scanBlockShared),sizeof(VarType)*::gridDimX));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&flag),sizeof(int)*::gridDimX));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&counter),sizeof(int)));

  CHECK(cudaMemset(scanBlockShared,static_cast<VarType>(0),sizeof(VarType)*::gridDimX));
  CHECK(cudaMemset(flag,0,sizeof(int)*::gridDimX));
  CHECK(cudaMemset(counter,0,sizeof(int)));

  for(size_t i=0;i<::N;++i)
    XCPU[i] = static_cast<::VarType>(i);
  
  /*Copy input data from CPU to GPU*/
  CHECK(cudaMemcpy(XGPU,XCPU.get(),sizeof(VarType)*::N,cudaMemcpyHostToDevice));

  auto start = std::chrono::high_resolution_clock::now();
  sequentialInclusiveScan(XCPU.get(),SSCPU.get(),::N);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpuTime = end-start;
  std::cout << "CPU time : " <<cpuTime.count()<<" ms"<<std::endl;

  dim3 block(::Nx,1,1);
  dim3 grid(::gridDimX,1,1);

  Timer timerSH("on the GPU with Kogge-Stone kernel and shuffle ");
  timerSH.start();
  inclusiveScanShuffleGPU<<<grid,block,sharedMemoryWarp>>>(XGPU,SSSHGPU,::N,flag,scanBlockShared,counter);
  CHECK(cudaDeviceSynchronize());
  timerSH.stop();
  CHECK(cudaGetLastError());
  
  CHECK(cudaMemcpy(SSSHGPUTest.get(),SSSHGPU,sizeof(VarType)*::N,cudaMemcpyDeviceToHost));

  std::cout<<"\n\n";
  for(size_t i=0;i<::N;++i){
    /*print a few last element*/
    if(i>N-10){
      std::cout<<"Idx = "<<i<<"\tSSCPU = "<<SSCPU[i]<<"\tSSDBGPU = "<<SSSHGPUTest[i]<<std::endl;
    }
    if(SSCPU[i]!=SSSHGPUTest[i]){
      std::cerr<<"Idx = "<<i<<"\tError : "<<"SSCPU = "<<SSCPU[i]<<"\tSSDBGPU = "<<SSSHGPUTest[i]<<std::endl;
      break;
    }
  }
  
  CHECK(cudaFree(XGPU));
  CHECK(cudaFree(SSSHGPU));
  CHECK(cudaFree(scanBlockShared));
  CHECK(cudaFree(flag));
  CHECK(cudaFree(counter));
  
  return EXIT_SUCCESS;
}
