/*
 * A custom deleter for cudaMalloc allocatoin
 Refereces:
 * [1] C++ Programming Language, Fourth Edition
 *
 * Programmed by: Kazem Bazesefidpar
 * Email: bazesefidpar64@gmail.com
 */
#include <iostream>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>
#include <utility>
#include <cstdlib>
#include <vector>
#include "./../../common/common.cuh"

/*Custom deleter for cudaMalloc allocatoin*/
auto deleterGPU = []<typename T>(T* ptr){if(ptr!=nullptr) CHECK(cudaFree(ptr));};

/*Using std::unique_ptr to manage the pointer (allocation and deallocation)*/
template<typename T>
std::unique_ptr<T,decltype(deleterGPU)> make_CUDAPtr(const size_t size){
  T* rawPtr{nullptr};
  CHECK(cudaMalloc(reinterpret_cast<void**>(&rawPtr), sizeof(T)*size));
  return std::unique_ptr<T, decltype(deleterGPU)>(rawPtr, deleterGPU);
}
/*Device function to add a value on the array on the GPU*/
template<typename T>
__global__ void add(T* ptr,const unsigned int N, const T value){
  unsigned int indx = blockIdx.x*blockDim.x+threadIdx.x;
  if(indx<N) ptr[indx]+=value;
}

int main(int argc,char* argv[]){
  const size_t N = 32; const double initialVal{100.0};
  /*Using std::unique_ptr to manage pointers*/
  std::unique_ptr<double[]> hostPtr = std::make_unique<double[]>(N);
  auto devicePtr = make_CUDAPtr<double>(N);
  /*Initialize data on both CPU and GPU*/
  std::for_each(hostPtr.get(),hostPtr.get()+N,[initialVal](auto& i){i=initialVal;});
  CHECK(cudaMemcpy(devicePtr.get(),hostPtr.get(),sizeof(double)*N,cudaMemcpyHostToDevice));
  /*Launch the kernel on the device*/
  add<<<32,1>>>(devicePtr.get(),N,5.0);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  /*Get back the results from GPU to CPU*/
  CHECK(cudaMemcpy(hostPtr.get(),devicePtr.get(),sizeof(double)*N,cudaMemcpyDeviceToHost));
  std::cout<<"Print 5 first elements : \n";
  std::for_each(hostPtr.get(),hostPtr.get()+5,[](const auto& i){std::cout<<i<<"\n";});
  
  return EXIT_SUCCESS;
}                         
