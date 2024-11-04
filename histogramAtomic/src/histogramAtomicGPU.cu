/* 
 * Kernels to compute the histogram of an input generated by random numbers.
 * 
 * The GPU kernel code is adapted from:
 * [1] "Programming Massively Parallel Processors: A Hands-on Approach 4th Edition" by David B. Kirk and Wen-mei W. Hwu, 
 * Chapter 9
 * 
 * Note: The code has been modified to suit the current application. 
 * Ensure to review the original source for detailed explanations.
 * 
 * Programmed by: Kazem Bazesefidpar
 * Email: bazesefidpar64@gmail.com
 */

#include <iostream>
#include <string>
#include <vector>
#include <type_traits>
#include <random>
#include <algorithm>
#include <memory>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "./../../common/common.cuh"

constexpr size_t N{1<<20};     /*Size of the data*/
constexpr size_t NX{1024};     /*Size of the thread block in CUDA programing model*/
constexpr float first{1.0f};   /*The range of random numbers generated : [first,last)*/ 
constexpr float last{10.0f};
constexpr size_t Divide{2};  /*The number of bin to divide the length "last-first"*/
constexpr bool plot{false};

/*A class to generate random numbers*/
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

/*Function to write the output*/
void write_histogram_to_file(const unsigned int* histogram, const size_t num_bins, const size_t divide,
			     const std::string& file_name) {
  std::ofstream outFile(file_name);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file for writing." << std::endl;
    return;
  }

  std::vector<std::string> bin_labels;

  for(size_t i=0;i<num_bins;++i){
    std::string input = std::to_string(static_cast<int>(i*divide))+"-"+
      std::to_string(static_cast<int>((i+1)*divide));
    bin_labels.push_back(input);
  }

  outFile << "Histogram Profile:\n";
  for (size_t i = 0; i < num_bins; ++i) {
    outFile << bin_labels[i] << ": " << histogram[i] << "\n";
  }
  outFile.close();
}
/*Function to compute the histogram on CPU
 input data        : data
 size of the data  : size
 divide            : the number of bins
 first             : the smallest number
 last              : the largest number
 histo             : histogram output
*/
template<typename T>
void histogramSequential(const T* data, const size_t size, const size_t divide,
			 const size_t first, const size_t last, unsigned int *histo){
  /*Compute the length of the bin*/
  size_t width = ((last-first)+divide-1)/divide;
  for (size_t i = 0; i < size; ++i) {
    int roundedVari = std::round(data[i]);
    /*check if the number is in the range [first,last]*/
    if (roundedVari >= static_cast<int>(first) && roundedVari <= static_cast<int>(last)) {
      size_t index = std::min(static_cast<size_t>(roundedVari / divide), width - 1);
      histo[index]++;
    }
  }
}
/*Function to compute the histogram on GPU with using the global memory
  The kernel is adapted from [1]
*/
template<typename T>
__global__ void histogramAtomicGPU(const T* data, const size_t size, const size_t divide,
                         const size_t first, const size_t last, unsigned int *histo){
  size_t width = ((last-first)+divide-1)/divide;
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(i<size) {
    int roundedVari = rintf(data[i]);
    if (roundedVari >= static_cast<int>(first) && roundedVari <= static_cast<int>(last)) {
      size_t r1 = static_cast<size_t>(roundedVari/divide);
      size_t r2 = width-1;
      size_t index = r1<r2 ? r1:r2;
      /*Use the atomic function to eliminate the race condition*/
      atomicAdd(&histo[index],1);
    }
  }
}
/*Kernel with privatization and using multiple output in the global memory
 The kernel is adapted from [1]
*/
template<typename T>
__global__ void histogramAtomicMOGPU(const T* data, const size_t size, const size_t divide,
                         const size_t first, const size_t last, unsigned int *histo){
  size_t width = ((last-first)+divide-1)/divide;
  size_t i     = blockIdx.x*blockDim.x+threadIdx.x;
  /*Compute the histogram in each thread block to decrease 
    the write contention to the same memory location */
  if(i<size) {
    int roundedVari = rintf(data[i]);
    if (roundedVari >= static_cast<int>(first) && roundedVari <= static_cast<int>(last)) {
      size_t r1 = static_cast<size_t>(roundedVari/divide);
      size_t r2 = width-1;
      size_t index = r1<r2 ? r1:r2;
      
      atomicAdd(&histo[blockIdx.x*width+index],1);
    }
  }
  /*Add the result of each thread block to the beginning of the histo[0:width]*/
  if(blockIdx.x>0 && threadIdx.x<width){
    __syncthreads();
    for(size_t j=threadIdx.x;j<width;j+=blockDim.x){
      unsigned int binValue = histo[blockIdx.x*width+j];
      if(binValue>0){
	atomicAdd(&histo[j],binValue);
      }
    }
  }
}
/*Kernel with privatization and using multiple output in the shared memory
  The kernel is adapted from [1]
*/
template<typename T>
__global__ void histogramAtomicMOSMGPU(const T* data, const size_t size, const size_t divide,
                         const size_t first, const size_t last, unsigned int *histo){
  size_t width = ((last-first)+divide-1)/divide;
  size_t i     = blockIdx.x*blockDim.x+threadIdx.x;
  
  extern __shared__ unsigned int histoTile[];
  
  for(size_t j=threadIdx.x;j<width;j+=blockDim.x){
    histoTile[j] = 0u;
  }
  __syncthreads();
  
  if(i<size) {
    int roundedVari = rintf(data[i]);
    if (roundedVari >= static_cast<int>(first) && roundedVari <= static_cast<int>(last)) {
      size_t r1 = static_cast<size_t>(roundedVari/divide);
      size_t r2 = width-1;
      size_t index = r1<r2 ? r1:r2;
      
      atomicAdd(&histoTile[index],1);
    }
  }
  __syncthreads();
  
  for(size_t j=threadIdx.x;j<width;j+=blockDim.x){
    unsigned int binValue = histoTile[j];
    if(binValue>0){
      atomicAdd(&histo[j],binValue);
    }
  }
  
}

int main(int argc, char* argv[]){

  int device = 0;
  CHECK(cudaSetDevice(device));

  size_t length = static_cast<size_t>(last-first);
  size_t width = (length+::Divide-1)/::Divide;
  dim3 block(::NX);
  dim3 grid((::N + block.x - 1) / block.x);
  
  size_t sharedMemory{width*sizeof(unsigned int)};
  /* Check if shared memory allocation exceeds device limits */
  {
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp,device));
    if(sharedMemory>iProp.sharedMemPerBlock){
      std::cout<<"The amount of the shared memory per block "<<iProp.sharedMemPerBlock<<
	", but the requeted amount is "<<sharedMemory<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  Random<float> randomFloat(first,last);
  std::unique_ptr<float[]> dataCPU       = std::make_unique<float[]>(::N);
  std::unique_ptr<unsigned int[]> histogramCPU = std::make_unique<unsigned int[]>(width);
  Timer timerCPU("on CPU");

  /*Data on the GPU for a single output*/
  float *dataGPU{nullptr};
  unsigned int *histogramGPU{nullptr};
  std::unique_ptr<unsigned int[]> histogramTest = std::make_unique<unsigned int[]>(width);
  Timer timerGPU("on GPU");

  /*Data on the GPU for privatization with multiple outputs*/
  unsigned int *histogramMOGPU{nullptr};
  Timer timerMOGPU("on MOGPU");
  std::unique_ptr<unsigned int[]> histogramMOTest = std::make_unique<unsigned int[]>(width);
  
  /*Data on the GPU for privatization with shared memory*/
  unsigned int *histogramMOSMGPU{nullptr};
  Timer timerMOSMGPU("on MOSHGPU");
  std::unique_ptr<unsigned int[]> histogramMOSMTest = std::make_unique<unsigned int[]>(width);
  
  CHECK(cudaMalloc(reinterpret_cast<void **>(&dataGPU),sizeof(float)*::N));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&histogramGPU),sizeof(unsigned int)*width));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&histogramMOGPU),sizeof(unsigned int)*width*grid.x));
  CHECK(cudaMalloc(reinterpret_cast<void **>(&histogramMOSMGPU),sizeof(unsigned int)*width));

  /*Initialize the input data*/
  std::for_each(dataCPU.get(),dataCPU.get()+::N,
 		[&randomFloat](auto &i){i=randomFloat.randomNumber();});
  /*Fill the output with zero on the CPU*/
  std::fill(histogramCPU.get(),histogramCPU.get()+width, 0);

  /*Fill the output with zero on the GPU*/
  CHECK(cudaMemcpy(dataGPU,dataCPU.get(),sizeof(float)*::N,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(histogramGPU,0,sizeof(unsigned int)*width));
  CHECK(cudaMemset(histogramMOGPU,0,sizeof(unsigned int)*width*grid.x));
  CHECK(cudaMemset(histogramMOSMGPU,0,sizeof(unsigned int)*width));

  /*Histogram computation on the CPU*/
  timerCPU.start();
  histogramSequential(dataCPU.get(),::N,::Divide,::first,::last,histogramCPU.get());
  timerCPU.stop();
  
  /*On GPU with a single copy of the output with global memory*/
  timerGPU.start();
  histogramAtomicGPU<<<grid,block>>>(dataGPU,::N,::Divide,::first,::last,histogramGPU);
  CHECK(cudaDeviceSynchronize());
  timerGPU.stop();
  CHECK(cudaGetLastError());
  
  /*On GPU with for privatization with multiple copy of the output in the global memory*/
  timerMOGPU.start();
  histogramAtomicMOGPU<<<grid,block>>>(dataGPU,::N,::Divide,::first,::last,histogramMOGPU);
  CHECK(cudaDeviceSynchronize());
  timerMOGPU.stop();
  CHECK(cudaGetLastError());

  /*On GPU with for privatization with multiple copy of the output in the shared memory*/
  timerMOSMGPU.start();
  histogramAtomicMOSMGPU<<<grid,block,sharedMemory>>>(dataGPU,::N,::Divide,::first,::last,histogramMOSMGPU);
  CHECK(cudaDeviceSynchronize());
  timerMOSMGPU.stop();
  CHECK(cudaGetLastError());

  /*Copy back the  output data from GPU to CPU*/
  CHECK(cudaMemcpy(histogramTest.get(),histogramGPU,sizeof(unsigned int)*width,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(histogramMOTest.get(),histogramMOGPU,sizeof(unsigned int)*width,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(histogramMOSMTest.get(),histogramMOSMGPU,sizeof(unsigned int)*width,cudaMemcpyDeviceToHost));
  
  for(size_t i=0;i<width;++i){
    if(histogramMOSMTest[i]!=histogramCPU[i]){
      std::cerr<<"Error["<<i<<"] = "<<histogramMOSMTest[i]<<"\t"<<histogramTest[i]<<"\n";
     break;
    }
  }
  
  if(::plot){
    write_histogram_to_file(histogramMOSMTest.get(),width,::Divide,"histogram_output.txt");
  }
  CHECK(cudaFree(dataGPU));
  CHECK(cudaFree(histogramGPU));
  CHECK(cudaFree(histogramMOGPU));
  CHECK(cudaFree(histogramMOSMGPU));
  
  return EXIT_SUCCESS;
}