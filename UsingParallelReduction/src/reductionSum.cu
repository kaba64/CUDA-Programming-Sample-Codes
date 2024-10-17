/*
 * Computation of Vector Length Using Parallel Sum Reduction Algorithm 
 *
 * This program computes the length of a vector with size N using various parallel algorithms.  
 *
 * The GPU kernel codes are adapted from the following references:     
 *
 * "Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition" by David B. Kirk and Wen-mei W. Hwu
 *
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/                                                             
 *
 * Note: The code has been modified to suit this application.                                                                 
 * Please review the original sources for detailed explanations.                                                                   
 *
 * Programmed by: Kazem Bazesefidpar
 * Email: bazesefidpar64@gmail.com
 *
 */
#include <iostream>
#include <type_traits>
#include <random>
#include <algorithm>
#include <memory>
#include <execution>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include "./../../common/common.cuh"

constexpr size_t N{1<<26};
constexpr size_t COARSE_FACTOR{2};
constexpr size_t NumThreads   = std::ceil(N/2); /*The total number of threads to process the data*/
constexpr size_t NumThreadsCF = std::ceil(N/(2*COARSE_FACTOR));
constexpr size_t NxShuffle{512};
constexpr size_t Nx{256};                      /*Number of threads in each block*/
using TypeVar = int;
constexpr TypeVar first{1};
constexpr TypeVar last{2};                 /*Generaating random number in [first,last)*/

template<typename T>
class Random{
private:
    T a, b;
    std::mt19937 generator;  // Mersenne Twister generator
public:
  Random(const T aIn = 0, const T bIn = 1)
    : a(aIn), b(bIn), generator(std::random_device{}()) {}  // Use random_device for seeding
  T randomNumber() {
    /*https://en.cppreference.com/w/cpp/types/is_floating_point*/
    if constexpr (std::is_floating_point<T>::value) {
      /*https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution*/
      /* Generate a random T in the range [a, b) */
      std::uniform_real_distribution<T> distribution(a, b);
      return distribution(generator);
    } else {
      /* Generate a random int in the range [a, b] */
      std::uniform_int_distribution<T> distribution(a, b);
      return distribution(generator);
    }
  }
};

template<typename T>
__global__ void sqrtDevice(T* x){
  *x = static_cast<T>(sqrt(static_cast<double>(*x)));
}
/*Kernnel with both control and memory divergence.
 It is adopted from references [4].*/
template<typename T>
__global__ void sumGPUKernel(T *input,T *output, const size_t nx){
  
  unsigned int i = 2*threadIdx.x+2*blockDim.x*blockIdx.x;
  for(unsigned int strid=1;strid<=blockDim.x;strid*=2){
    if(threadIdx.x%strid==0 && i+strid<nx){
      input[i]+=input[i+strid];
    }
   __syncthreads();
  }
  if(threadIdx.x==0){
    atomicAdd(output,input[2*blockIdx.x*blockDim.x]);
  }
}
/*Kernnel with improved control and memory divergence.
 It is adopted from references [4].*/
template<typename T>
__global__ void sumGPUKernelCD(T *input, T *output, const size_t nx,const size_t factor){

  unsigned int i = factor*2*blockIdx.x*blockDim.x+threadIdx.x;
  
  T reduction;
  if(i<nx)
    reduction = input[i];
  
  for(unsigned int strid=1;strid<factor*2;++strid){
    if(i+strid*blockDim.x<nx){
      reduction+=input[i+strid*blockDim.x];
    }
  }
  if(i<nx)
    input[i]=reduction;
  __syncthreads();
  
  for(unsigned int strid=blockDim.x/2;strid>0;strid/=2){
    if(threadIdx.x<strid && i+strid<nx){
      input[i]+=input[i+strid];
    }
   __syncthreads();
  }
  if(threadIdx.x==0){
    atomicAdd(output,input[factor*2*blockIdx.x*blockDim.x]);
  }
}
/* Kernel using shared memory and warp-level primitives
 * with improved control divergence and memory access patterns.
 * Adapted from references [3] and [4]. */
template<typename T>
__global__ void sumGPUKernelCDSM(T *input,T *output,const size_t nx,const size_t factor){

  extern __shared__ T dataSM[];
  unsigned int i = factor*2*blockIdx.x*blockDim.x+threadIdx.x;
  T reduction;
  if(i<nx)
    reduction = input[i];
  for(unsigned int strid=1;strid<factor*2;++strid){
    if(i+strid*blockDim.x<nx){
      reduction+=input[i+strid*blockDim.x];
    }
  }
  if(i<nx)
    dataSM[threadIdx.x]=reduction;
  __syncthreads();
  for(unsigned int strid=blockDim.x/2;strid>=warpSize;strid/=2){
    if(threadIdx.x<strid && i+strid<nx){
      dataSM[threadIdx.x]+=dataSM[threadIdx.x+strid];
    }
   __syncthreads();
  }
  T data{0};
  unsigned mask = __ballot_sync(0xffffffff,i<nx);
  if (threadIdx.x <warpSize){
    data = dataSM[threadIdx.x];
    for(unsigned int j=warpSize/2;j>0;j/=2)
      data+=__shfl_down_sync(0xffffffff,data,j);
  }
  if(threadIdx.x==0){
    atomicAdd(output,data);
  }
}    
/*Kernel using warp-level priitives.
  It is adopted from references [3]*/
template<typename T>
__global__ void reductionShuffle(T *input,T *output,const size_t nx){

  extern __shared__ T partialSumSM[];
  unsigned int idx{threadIdx.x%warpSize};
  unsigned int idxG{threadIdx.x/warpSize};
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  T var{input[i]};
  unsigned mask = __ballot_sync(0xffffffff,i<nx);
  for(unsigned int j=warpSize/2;j>0;j/=2)
    var+=__shfl_down_sync(mask,var,j);
  
  if(idx==0){
    partialSumSM[idxG] = var;
  }
  __syncthreads();
  
  if(threadIdx.x<(blockDim.x/warpSize))
    var = partialSumSM[threadIdx.x];
  else if(threadIdx.x<warpSize)
    var=0;
  
  if(idxG==0){
    for(unsigned int j=warpSize/2;j>0;j/=2)
    var+=__shfl_down_sync(mask,var,j);
  }
  
  if(threadIdx.x==0){
    atomicAdd(output,var);
  }
}
template<typename T>
__global__ void multiplication(T *input,const size_t nx){
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  if(i<nx)
    input[i]*=input[i];
}

int main(int argc, char* argv[]){

  int device = 0;
  CHECK(cudaSetDevice(device));
  
  /*Calculate the size of the shared memory for the each thread block*/
  size_t sharedMemory{::Nx*sizeof(TypeVar)};
  {
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp,device));
    if(sharedMemory>iProp.sharedMemPerBlock){
      std::cout<<"The amount of the shared memory per block "<<iProp.sharedMemPerBlock
	       <<", but the requeted amount is "<<sharedMemory<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
  size_t sharedMemorySuffle = std::ceil(::NxShuffle/32)*sizeof(TypeVar);
  {
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp,device));
    if(sharedMemorySuffle>iProp.sharedMemPerBlock){
      std::cout<<"The amount of the shared memory per block "<<iProp.sharedMemPerBlock
               <<", but the requeted amount is "<<sharedMemorySuffle<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  Random<TypeVar> randomInt(::first,::last);
  /*CPU allocation*/
  std::unique_ptr<TypeVar[]> dataCPU = std::make_unique<TypeVar[]>(::N);
  TypeVar sumCPU{0};
  
  /*GPU allocation with control divergence*/
  TypeVar *dataGPU{nullptr}, *sumGPU{nullptr};
  TypeVar sumGPUTest{0};

  /*GPU decleration with improved control divergence*/
  TypeVar *dataCDGPU{nullptr}, *sumCDGPU{nullptr};
  TypeVar sumCDGPUTest{0};

  /*GPU decleration with improved control divergence and shared memory*/
  TypeVar *dataCDSMGPU{nullptr}, *sumCDSMGPU{nullptr};
  TypeVar sumCDSMGPUTest{0};

  /*GPU decleration for shuffle*/
  TypeVar *dataShuffleGPU{nullptr}, *sumShuffleGPU{nullptr};
  TypeVar sumShuffleGPUTest{0};
  /*Data allocatin on the GPU*/
  
  /*Initialize on the CPU*/
  std::for_each(std::execution::par,dataCPU.get(),dataCPU.get()+::N,
                [&randomInt](auto &i){i=randomInt.randomNumber();});
  
  /*Data allocatin on the GPU*/
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dataGPU),sizeof(TypeVar)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&sumGPU),sizeof(TypeVar)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dataCDGPU),sizeof(TypeVar)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&sumCDGPU),sizeof(TypeVar)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dataCDSMGPU),sizeof(TypeVar)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&sumCDSMGPU),sizeof(TypeVar)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dataShuffleGPU),sizeof(TypeVar)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&sumShuffleGPU),sizeof(TypeVar)));

  /*Copy data to GPU*/
  CHECK(cudaMemcpy(dataGPU,dataCPU.get(),sizeof(TypeVar)*::N,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dataCDGPU,dataCPU.get(),sizeof(TypeVar)*::N,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dataCDSMGPU,dataCPU.get(),sizeof(TypeVar)*::N,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dataShuffleGPU,dataCPU.get(),::N*sizeof(TypeVar),cudaMemcpyHostToDevice));
  
  /*Coputation on the CPU*/
  /*https://en.cppreference.com/w/cpp/algorithm/reduce*/
  auto start = std::chrono::high_resolution_clock::now();
  std::for_each(std::execution::par,dataCPU.get(),dataCPU.get()+::N,
                [](auto &i){i*=i;});
  sumCPU = std::reduce(std::execution::par, dataCPU.get(), dataCPU.get()+::N);
  sumCPU = std::sqrt(sumCPU);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpuTime = end-start;
  std::cout << "CPU time : " <<cpuTime.count()<<" ms"<<std::endl;
  
  dim3 block(::Nx,1,1);
  dim3 grid((::NumThreads+block.x-1)/block.x,1,1);
  dim3 gridCF((::NumThreadsCF+block.x-1)/block.x,1,1);
  
  Timer timerSimple("on the GPU with control divergence");
  timerSimple.start();
  multiplication<<<(::N + ::Nx-1)/::Nx,::Nx>>>(dataGPU,::N);
  CHECK(cudaDeviceSynchronize());
  sumGPUKernel<<<grid,block>>>(dataGPU,sumGPU,::N);
  CHECK(cudaDeviceSynchronize());
  sqrtDevice<<<1,1>>>(sumGPU);
  CHECK(cudaDeviceSynchronize());
  timerSimple.stop();
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(&sumGPUTest,sumGPU,sizeof(TypeVar),cudaMemcpyDeviceToHost));
  
  Timer timerCD("on the GPU with improved CD");
  timerCD.start();
  multiplication<<<(::N + ::Nx-1)/::Nx,::Nx>>>(dataCDGPU,::N);
  CHECK(cudaDeviceSynchronize());
  sumGPUKernelCD<<<gridCF,block>>>(dataCDGPU,sumCDGPU,::N,::COARSE_FACTOR);
  CHECK(cudaDeviceSynchronize());
  sqrtDevice<<<1,1>>>(sumCDGPU);
  CHECK(cudaDeviceSynchronize());
  timerCD.stop();
  CHECK(cudaGetLastError());
    
  CHECK(cudaMemcpy(&sumCDGPUTest,sumCDGPU,sizeof(TypeVar),cudaMemcpyDeviceToHost));
  
  Timer timerCDSM("on the GPU with improved CDSH");
  timerCDSM.start();
  multiplication<<<(::N + ::Nx-1)/::Nx,::Nx>>>(dataCDSMGPU,::N);
  CHECK(cudaDeviceSynchronize());
  sumGPUKernelCDSM<<<gridCF,block,sharedMemory>>>(dataCDSMGPU,sumCDSMGPU,::N,::COARSE_FACTOR);
  CHECK(cudaDeviceSynchronize());
  sqrtDevice<<<1,1>>>(sumCDSMGPU);
  CHECK(cudaDeviceSynchronize());
  timerCDSM.stop();
  CHECK(cudaGetLastError());
  
  CHECK(cudaMemcpy(&sumCDSMGPUTest,sumCDSMGPU,sizeof(TypeVar),cudaMemcpyDeviceToHost));
  
  dim3 blockShuffle(NxShuffle,1,1);
  dim3 gridShuffle((::N+blockShuffle.x-1)/blockShuffle.x,1,1);
  Timer timerShuffle("on the GPU with shuffle funnction");
  timerShuffle.start();
  multiplication<<<(::N + ::Nx-1)/::Nx,::Nx>>>(dataShuffleGPU,::N);
  CHECK(cudaDeviceSynchronize());
  reductionShuffle<<<gridShuffle,blockShuffle,sharedMemorySuffle>>>(dataShuffleGPU,sumShuffleGPU,::N);
  CHECK(cudaDeviceSynchronize());
  sqrtDevice<<<1,1>>>(sumShuffleGPU);
  CHECK(cudaDeviceSynchronize());
  timerShuffle.stop();
  CHECK(cudaGetLastError());
  
  CHECK(cudaMemcpy(&sumShuffleGPUTest,sumShuffleGPU,sizeof(TypeVar),cudaMemcpyDeviceToHost));
  
  if(std::abs(sumCDSMGPUTest-sumCPU)>0.01 || std::abs(sumCDGPUTest-sumCPU)>0.01 ||
    std::abs(sumShuffleGPUTest-sumCPU)>0.01){
    std::cout<<std::setprecision(20)<<"\t sumGPUTest = "<<sumGPUTest
	     <<"\t sumCDGPUTest = "<<sumCDGPUTest<<"\t sumCDSMGPUTest = "<<sumCDSMGPUTest<<
      "\t sumShuffleGPUTest = "<<sumShuffleGPUTest<<"\tsumCPU = "<<sumCPU<<std::endl;
  }
  cudaFree(dataGPU);
  cudaFree(sumGPU);
  cudaFree(dataCDGPU);
  cudaFree(sumCDGPU);
  cudaFree(dataCDSMGPU);
  cudaFree(sumCDSMGPU);
  cudaFree(dataShuffleGPU);
  cudaFree(sumShuffleGPU);
  return EXIT_SUCCESS;
}
