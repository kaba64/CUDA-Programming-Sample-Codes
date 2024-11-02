/*
 * Applying 2D Convolution on Image Channels using Gaussian Blur Filter
 *
 * This program loads an image and computes its convolution using a Gaussian Blur filter.
 * It performs this computation on both the CPU and GPU, utilizing shared and constant memory
 * in the CUDA programming model for optimal performance.
 *
 * Some GPU kernel codes are adapted from the following reference:
 *
 * [1] "Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition" 
 *     by David B. Kirk and Wen-mei W. Hwu
 *
 * Note: This code has been modified for this specific application.
 * Please refer to the original sources for detailed explanations.
 *
 * Programmed by: Kazem Bazesefidpar
 * Email: bazesefidpar64@gmail.com
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "./../../common/common.cuh"

constexpr size_t RADIUS{2};          /*Filter radius*/
constexpr size_t Nx{32};             /*Block sizes in x and y-directions*/
constexpr size_t Ny{32};
constexpr bool PLOT{true};
using TypeVar = float;
#ifdef DEBUG
constexpr TypeVar epsError{0.001}; /*Maximum differnce between CPU and GPU results*/
#endif

template<typename T>
void fillChannelData(const std::vector<cv::Mat>& channels, const size_t nx, const size_t ny,
		     const size_t channelIndex, std::vector<T>& data){
  if(channelIndex>(channels.size()-1)){
    std::cerr<<"Invalid size for channel"<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  for (size_t j = 0; j <ny; ++j) {
    for (size_t i = 0; i<nx; ++i) {
      data[j*nx + i] = static_cast<T>(channels[channelIndex].at<uchar>(j,i));
    }
  }
}

__constant__ TypeVar filterConst[(2*::RADIUS+1)*(2*::RADIUS+1)];
/*Kernel using the global memory.                
  It is adopted from reference [1].*/
template<typename T>
__global__ void computeConvolutionGPU(const T* data, T* dataOut,const size_t radius,
				      const size_t nx, const size_t ny){

  unsigned int HX = static_cast<unsigned int>(2*radius+1);
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;
  
  if (i < nx && j < ny){
    T sum = static_cast<T>(0);
    for (int y = -static_cast<int>(radius); y <= static_cast<int>(radius); ++y) {
      for (int x = -static_cast<int>(radius); x <= static_cast<int>(radius); ++x) {
	int ii = i + x;
	int jj = j + y;
	if (ii >= 0 && ii < static_cast<int>(nx) && jj >= 0 && jj < static_cast<int>(ny)){
	  /*Apply zero boundary condition*/
	  sum += data[static_cast<size_t>(jj*nx+ii)]*filterConst[static_cast<size_t>((y+radius)*HX+(x+radius))];
	}
      }
    }
    dataOut[j*nx+i] = sum;
  }
}
/*Kernel using the shared memory with the same size of the launched thread block.
  It is adopted from reference [1].*/
template<typename T>
__global__ void computeConvolutionSMIGPU(const T* data, T* dataOut,const size_t radius,
                                      const size_t nx, const size_t ny){
  extern __shared__ T dataSM[];
  
  int tx  = static_cast<int>(threadIdx.x-radius);
  int ty  = static_cast<int>(threadIdx.y-radius);
  unsigned int HX     = static_cast<unsigned int>(2*radius+1);
  unsigned int DIMX   = static_cast<unsigned int>(blockDim.x-2*radius);
  unsigned int DIMY   =	static_cast<unsigned int>(blockDim.y-2*radius);
  int i = DIMX*blockIdx.x+threadIdx.x-radius;
  int j = DIMY*blockIdx.y+threadIdx.y-radius;
  
  /*Upload the points in the thread block into the shared memory
    apply zero boundary condition*/
  dataSM[threadIdx.y*blockDim.x+threadIdx.x] = (i>=0 && i<nx && j>=0 && j<ny) ?
    data[j*nx+i]:static_cast<T>(0);
  
  __syncthreads();

  if(i>=0 && i<nx && j>=0 && j<ny){
    if(tx>=0 && tx<DIMX &&
       ty>=0 && ty<DIMY){
      T sum = static_cast<T>(0);
      for (int y = -static_cast<int>(radius); y <= static_cast<int>(radius); ++y) {
	for (int x = -static_cast<int>(radius); x <= static_cast<int>(radius); ++x) {
	  sum += dataSM[static_cast<size_t>((threadIdx.y+y)*blockDim.x+(threadIdx.x+x))]*
	    filterConst[static_cast<size_t>((y+radius)*HX+(x+radius))];
	}
      }
      dataOut[j*nx+i] = sum;
    }
  }
}      
/*Kernel using the shared memory with the larger size of the launched thread block.*/
template<typename T>
__global__ void computeConvolutionSMIIGPU(const T* data, T* dataOut,const size_t radius,
                                      const size_t nx, const size_t ny){

  extern __shared__ T dataSM[];
  unsigned int SMDimX = static_cast<unsigned int>(blockDim.x+2*radius);
  unsigned int tx     = static_cast<unsigned int>(threadIdx.x+radius);
  unsigned int ty     = static_cast<unsigned int>(threadIdx.y+radius);
  unsigned int HX     = static_cast<unsigned int>(2*radius+1);
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;
  
  /*Upload the points in the thread block into the shared memory
    apply zero boundary condition*/
  dataSM[ty*SMDimX+tx] = (i<nx && j < ny) ? data[j*nx+i] : static_cast<T>(0);
  
  /*Upload the halo cells in x-dirction with threadIdx.x=0
   and use the threads in a row for the efficient use of resources in a warp*/
  unsigned int ROW = blockDim.y>2 ? 1:0;
  if(threadIdx.x<blockDim.x && threadIdx.y==ROW){
    for(unsigned int threadY=threadIdx.x;threadY<blockDim.y;threadY+=blockDim.x){
      unsigned int iIn  = blockDim.x*blockIdx.x+0;
      unsigned int jIn  = blockDim.y*blockIdx.y+threadY;
      unsigned int tyIn = static_cast<unsigned int>(threadY+radius);
      for(unsigned int filterRadius=0;filterRadius<radius;++filterRadius){
	dataSM[tyIn*SMDimX+(0+filterRadius)] =
	  (static_cast<int>((iIn+filterRadius-radius))>=0 && (iIn+filterRadius-radius)<nx && jIn<ny)?
	  data[jIn*nx+(iIn+filterRadius-radius)] : static_cast<T>(0);  /*threadIdx.x = 0*/
      }
    }
  }
  /*Upload the halo cells in x-dirction with threadIdx.x=blockDim.x-1
   and use the threads in a row for the efficient use of resources in a warp*/
  ROW = blockDim.y>3 ? 2:ROW;
  if(threadIdx.x<blockDim.x && threadIdx.y==ROW){
    for(unsigned int threadY=threadIdx.x;threadY<blockDim.y;threadY+=blockDim.x){
      unsigned int iIn  = blockDim.x*blockIdx.x+(blockDim.x-1);
      unsigned int jIn  = blockDim.y*blockIdx.y+threadY;
      unsigned int txIn = static_cast<unsigned int>((blockDim.x-1)+radius);
      unsigned int tyIn = static_cast<unsigned int>(threadY+radius);
      for(unsigned int filterRadius=1;filterRadius<=radius;++filterRadius){
	dataSM[tyIn*SMDimX+(txIn+filterRadius)] = ((iIn+filterRadius)<nx && jIn<ny) ?
	  data[jIn*nx+iIn+filterRadius] : static_cast<T>(0); /*threadIdx.x = blockDim.x-1*/
      }
    }
  }
  /*Upload the halo cells y-direcion with threadIdx.y=0*/
  if(threadIdx.y==0){
    for(unsigned int filterRadius=0;filterRadius<radius;++filterRadius){
      dataSM[(threadIdx.y+filterRadius)*SMDimX+tx] = (i<nx && static_cast<int>((j+filterRadius-radius))>=0)?
  	data[(j+filterRadius-radius)*nx+i]:static_cast<T>(0);
    }
  }
  /*Upload the halo cells y-direcion with threadIdx.y=blockDim.y-1*/
  if(threadIdx.y==(blockDim.y-1)){
    for(unsigned int filterRadius=1;filterRadius<=radius;++filterRadius){
      dataSM[(ty+filterRadius)*SMDimX+tx] = (i<nx && (j+filterRadius)<ny) ?
	data[(j+filterRadius)*nx+i] : static_cast<T>(0);
    }
  }
  /*Upload the top left corner halo cells*/
  if(threadIdx.x==0 && threadIdx.y==0){
    for(unsigned int filterRadius=0;filterRadius<radius;++filterRadius){
      for(unsigned int StepX=0;StepX<radius;++StepX){
	dataSM[(threadIdx.y+filterRadius)*SMDimX+threadIdx.x+StepX] =
	  (static_cast<int>((j+filterRadius-radius))>=0 && static_cast<int>(i+StepX-radius)>=0) ?
	  data[(j+filterRadius-radius)*nx+i+StepX-radius] : static_cast<T>(0);
      }
    }
  }
  /*Upload the top right corner halo cells*/
  if(((threadIdx.x==(blockDim.x-1) || i==nx-1) && threadIdx.y==0)){
    for(unsigned int filterRadius=0;filterRadius<radius;++filterRadius){
      for(unsigned int StepX=1;StepX<=radius;++StepX){
	dataSM[(threadIdx.y+filterRadius)*SMDimX+tx+StepX] =
	  ((i+StepX)<nx && static_cast<int>(j+filterRadius-radius)>=0) ?
	  data[(j+filterRadius-radius)*nx+i+StepX] : static_cast<T>(0);
      }
    }
  }
  /*Upload the bottom left corner halo cells*/
  if(threadIdx.x==0 && threadIdx.y==(blockDim.y-1)){
    for(unsigned int filterRadius=1;filterRadius<=radius;++filterRadius){
      for(unsigned int StepX=0;StepX<radius;++StepX){
	dataSM[(ty+filterRadius)*SMDimX+threadIdx.x+StepX] =
	  (static_cast<int>(i+StepX-radius)>=0 && (j+filterRadius)<ny) ?
	  data[(j+filterRadius)*nx+i+StepX-radius] : static_cast<T>(0);
      }
    }
  }
  /*Upload the bottom right corner halo cells*/
  if(threadIdx.x==(blockDim.x-1) && threadIdx.y==(blockDim.y-1)){
    for(unsigned int filterRadius=1;filterRadius<=radius;++filterRadius){
      for(unsigned int StepX=1;StepX<=radius;++StepX){
	dataSM[(ty+filterRadius)*SMDimX+tx+StepX] = ((i+StepX)<nx && (j+filterRadius)<ny) ?
	  data[(j+filterRadius)*nx+i+StepX] : static_cast<T>(0);
      }
    }
  }
  __syncthreads();
  if (i<nx && j < ny){
    T sum = static_cast<T>(0);
    for (int y = -static_cast<int>(radius); y <= static_cast<int>(radius); ++y) {
      for (int x = -static_cast<int>(radius); x <= static_cast<int>(radius); ++x) {
	sum += dataSM[static_cast<size_t>((ty+y)*SMDimX+(tx+x))]*
	  filterConst[static_cast<size_t>((y+radius)*HX+(x+radius))];
      }
    }
    dataOut[j*nx+i] = sum;
  }
}
/*Kernel using the shared memory with the same size of the launched thread block and possible usage of L2 cache.
  It is adopted from reference [1].*/
template<typename T>
__global__ void computeConvolutionSMCacheGPU(const T* data, T* dataOut,const size_t radius,
                                      const size_t nx, const size_t ny){
  extern __shared__ T dataSM[];
  
  unsigned int HX     = static_cast<unsigned int>(2*radius+1);
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;
  
  /*Upload the points in the thread block into the shared memory
    apply zero boundary condition*/
  dataSM[threadIdx.y*blockDim.x+threadIdx.x] = (i<nx && j<ny) ? data[j*nx+i] : static_cast<T>(0);
  
  __syncthreads();
  
  if(i<nx && j<ny){
    T sum = static_cast<T>(0);
    for (int y = -static_cast<int>(radius); y <= static_cast<int>(radius); ++y) {
      for (int x = -static_cast<int>(radius); x <= static_cast<int>(radius); ++x) {
	if((i+x)>=(blockDim.x*blockIdx.x) && (i+x)<(blockDim.x*(blockIdx.x+1)) &&
	   (j+y)>=(blockDim.y*blockIdx.y) && (j+y)<(blockDim.y*(blockIdx.y+1))){
	  sum += dataSM[static_cast<size_t>((threadIdx.y+y)*blockDim.x+(threadIdx.x+x))]*
	    filterConst[static_cast<size_t>((y+radius)*HX+(x+radius))]; 
	}else if(static_cast<int>(i+x)>=0 && (i+x)<nx &&
		 static_cast<int>(j+y)>=0 && (j+y)<ny){
	  sum += data[(j+y)*nx+(i+x)]*filterConst[static_cast<size_t>((y+radius)*HX+(x+radius))];
	}
      }
    }
    dataOut[j*nx+i] = sum;
  }
}

template<typename T>
void computeConvolution(const std::vector<T>& data, const std::vector<T>& filter, const size_t radius,
                        const size_t nx, const size_t ny, std::vector<T>& dataOut) {

  unsigned int HX = 2*radius+1;
  
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      T sum{};
      for (int y = -static_cast<int>(radius); y <= static_cast<int>(radius); ++y) {
	for (int x = -static_cast<int>(radius); x <= static_cast<int>(radius); ++x) {
	  int ii = i + x;
	  int jj = j + y;
	  if (ii >= 0 && ii < static_cast<int>(nx) && jj >= 0 && jj < static_cast<int>(ny)){
	    /*Apply zero boundary condition*/
	    sum += data[static_cast<size_t>(jj*nx+ii)] * filter[static_cast<size_t>((y+radius)*HX+(x+radius))];
	  }
	}
      }
      dataOut[j*nx+i] = sum;
    }
  }
}


int main(int argc,char* argv[]){
  
  int device = 0;
  CHECK(cudaSetDevice(device));
  
  /* Load color image */
  cv::Mat image = cv::imread("sample.jpg", cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Image not found!" << std::endl;
    return EXIT_FAILURE;
  }
  
  /* Split the image into BGR channels */
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  
  /*Define a Gaussian Blur filter as a 1D vector and normalize it*/
  std::vector<TypeVar> filter = {
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
  };
  for (size_t i = 0; i < filter.size(); ++i) {
    filter[i] /= 256.0f;
  }
  /*Copy filter to constant memory*/
  {
    size_t constantMem{(2*::RADIUS+1)*(2*::RADIUS+1)*sizeof(TypeVar)};
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, device));
    if(deviceProp.totalConstMem<constantMem){
      std::cerr<<"The size of the constant memory ("<<deviceProp.totalConstMem<<
  	") is smaller than the requsted ammount "<<constantMem<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
   /*Calculate the size of the shared memory for the each thread block*/
  size_t sharedMemoryI{::Nx*::Ny*sizeof(TypeVar)};
  size_t sharedMemoryII{(::Nx+2*::RADIUS)*(::Ny+2*::RADIUS)*sizeof(TypeVar)};
  {
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp,device));
    if(sharedMemoryII>iProp.sharedMemPerBlock){
      std::cout<<"The amount of the shared memory per block for II"<<iProp.sharedMemPerBlock
	       <<", but the requeted amount is "<<sharedMemoryII<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
  /*Copy the filter to the constant memory on the GPU */
  CHECK(cudaMemcpyToSymbol(filterConst,filter.data(),(2*::RADIUS+1)*(2*::RADIUS+1)*sizeof(TypeVar)));

  size_t nx = static_cast<size_t>(image.cols);
  size_t ny = static_cast<size_t>(image.rows);
  
  /*Vectors to hold flattened channel data */
  std::vector<TypeVar> Bdata(nx*ny,0.0f);
  std::vector<TypeVar> Gdata(nx*ny,0.0f);
  std::vector<TypeVar> Rdata(nx*ny,0.0f);
  
  /*Allocate data on the GPU for inputs*/
  TypeVar *BdataGPU{nullptr}, *GdataGPU{nullptr}, *RdataGPU{nullptr};
  
  CHECK(cudaMalloc(reinterpret_cast<void**>(&BdataGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&GdataGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&RdataGPU),sizeof(TypeVar)*nx*ny));
  
  /* Fill the channel data using the fillChannelData function */
  fillChannelData(channels,nx,ny,0,Bdata);
  fillChannelData(channels,nx,ny,1,Gdata); 
  fillChannelData(channels,nx,ny,2,Rdata);
  
  /*Copy input data to GPU memory */
  CHECK(cudaMemcpy(BdataGPU,Bdata.data(),sizeof(TypeVar)*nx*ny,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(GdataGPU,Gdata.data(),sizeof(TypeVar)*nx*ny,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(RdataGPU,Rdata.data(),sizeof(TypeVar)*nx*ny,cudaMemcpyHostToDevice));
  
#ifdef DEBUG
  /*Applied convolution on CPU data*/
  std::vector<TypeVar> BConvCPU(nx*ny,0.0f);
  std::vector<TypeVar> GConvCPU(nx*ny,0.0f);
  std::vector<TypeVar> RConvCPU(nx*ny,0.0f);
  
  auto start = std::chrono::high_resolution_clock::now();
  /*Compute the convolution of each channel by Gaussian Blur filter on CPU*/
  computeConvolution(Bdata,filter,::RADIUS,nx,ny,BConvCPU);
  computeConvolution(Gdata,filter,::RADIUS,nx,ny,GConvCPU);
  computeConvolution(Rdata,filter,::RADIUS,nx,ny,RConvCPU);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpuTime = end-start;
  std::cout << "CPU time : " <<cpuTime.count()<<" ms"<<std::endl;
#endif
  
  /*Variables to hold outputs computed using global memory on the GPU*/
  TypeVar *BdataOutGPU{nullptr}, *GdataOutGPU{nullptr}, *RdataOutGPU{nullptr};
  std::vector<TypeVar> BGPU(nx*ny,0.0f), GGPU(nx*ny,0.0f), RGPU(nx*ny,0.0f);

  /*Variables to hold outputs computed using shared memory I on the GPU*/
  TypeVar *BdataOutSMIGPU{nullptr}, *GdataOutSMIGPU{nullptr}, *RdataOutSMIGPU{nullptr};
  std::vector<TypeVar> BSMIGPU(nx*ny,0.0f), GSMIGPU(nx*ny,0.0f), RSMIGPU(nx*ny,0.0f);
  
  /*Variables to hold outputs computed using shared memory II on the GPU*/
  TypeVar *BdataOutSMIIGPU{nullptr}, *GdataOutSMIIGPU{nullptr}, *RdataOutSMIIGPU{nullptr};
  std::vector<TypeVar> BSMIIGPU(nx*ny,0.0f), GSMIIGPU(nx*ny,0.0f), RSMIIGPU(nx*ny,0.0f);

  /*Variables to hold outputs computed using shared memory and probably L2 cache on the GPU*/
  TypeVar *BdataOutSMCacheGPU{nullptr}, *GdataOutSMCacheGPU{nullptr}, *RdataOutSMCacheGPU{nullptr};
  std::vector<TypeVar> BSMCacheGPU(nx*ny,0.0f), GSMCacheGPU(nx*ny,0.0f), RSMCacheGPU(nx*ny,0.0f);
  
  /*Allocate data on the GPU for outputs*/
  CHECK(cudaMalloc(reinterpret_cast<void**>(&BdataOutGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&GdataOutGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&RdataOutGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&BdataOutSMIGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&GdataOutSMIGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&RdataOutSMIGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&BdataOutSMIIGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&GdataOutSMIIGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&RdataOutSMIIGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&BdataOutSMCacheGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&GdataOutSMCacheGPU),sizeof(TypeVar)*nx*ny));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&RdataOutSMCacheGPU),sizeof(TypeVar)*nx*ny));
  
  /*Set the block and grid dimension*/
  dim3 block(::Nx,::Ny,1);
  dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y,1);
  /*Set the block and grid dimension for shared memory algorithm I*/
  dim3 blockI(::Nx-2*::RADIUS,::Ny-2*::RADIUS,1);
  dim3 gridI((nx+blockI.x-1)/blockI.x,(ny+blockI.y-1)/blockI.y,1);
  
  Timer timerGM("on the GPU with global memory");
  size_t NumIter{5};
  for(size_t Iter=0;Iter<NumIter;++Iter){
    timerGM.start();
    /*Compute the convolution on GPU with global memory*/
    computeConvolutionGPU<<<grid,block>>>(BdataGPU,BdataOutGPU,::RADIUS,nx,ny);
    computeConvolutionGPU<<<grid,block>>>(GdataGPU,GdataOutGPU,::RADIUS,nx,ny);
    computeConvolutionGPU<<<grid,block>>>(RdataGPU,RdataOutGPU,::RADIUS,nx,ny);
    CHECK(cudaDeviceSynchronize());
    timerGM.stop(false);
    CHECK(cudaGetLastError());
  }
  timerGM.averageTime();
  
  /*Copy output data computed using global memory from GPU to CPU*/
  CHECK(cudaMemcpy(BGPU.data(),BdataOutGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(GGPU.data(),GdataOutGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(RGPU.data(),RdataOutGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));

  Timer timerSMI("on the GPU with shared memory I");
  for(size_t Iter=0;Iter<NumIter;++Iter){
    timerSMI.start();
    /*Compute the convolution on GPU with shared memory II*/
    computeConvolutionSMIGPU<<<gridI,block,sharedMemoryI>>>(BdataGPU,BdataOutSMIGPU,::RADIUS,nx,ny);
    computeConvolutionSMIGPU<<<gridI,block,sharedMemoryI>>>(GdataGPU,GdataOutSMIGPU,::RADIUS,nx,ny);
    computeConvolutionSMIGPU<<<gridI,block,sharedMemoryI>>>(RdataGPU,RdataOutSMIGPU,::RADIUS,nx,ny);
    CHECK(cudaDeviceSynchronize());
    timerSMI.stop(false);
    CHECK(cudaGetLastError());
  }
  timerSMI.averageTime();
  
  /*Copy output data computed using shared memory from GPU to CPU*/
  CHECK(cudaMemcpy(BSMIGPU.data(),BdataOutSMIGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(GSMIGPU.data(),GdataOutSMIGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(RSMIGPU.data(),RdataOutSMIGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  
  Timer timerSMII("on the GPU with shared memory II");
  for(size_t Iter=0;Iter<NumIter;++Iter){
    timerSMII.start();
    /*Compute the convolution on GPU with shared memory II*/
    computeConvolutionSMIIGPU<<<grid,block,sharedMemoryII>>>(BdataGPU,BdataOutSMIIGPU,::RADIUS,nx,ny);
    computeConvolutionSMIIGPU<<<grid,block,sharedMemoryII>>>(GdataGPU,GdataOutSMIIGPU,::RADIUS,nx,ny);
    computeConvolutionSMIIGPU<<<grid,block,sharedMemoryII>>>(RdataGPU,RdataOutSMIIGPU,::RADIUS,nx,ny);
    CHECK(cudaDeviceSynchronize());
    timerSMII.stop(false);
    CHECK(cudaGetLastError());
  }
  timerSMII.averageTime();
  
  /*Copy output data computed using shared memory from GPU to CPU*/
  CHECK(cudaMemcpy(BSMIIGPU.data(),BdataOutSMIIGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(GSMIIGPU.data(),GdataOutSMIIGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(RSMIIGPU.data(),RdataOutSMIIGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  
  Timer timerSMCache("on the GPU with shared memory and probably L2 cache");
  for(size_t Iter=0;Iter<NumIter;++Iter){
    timerSMCache.start();
    /*Compute the convolution on GPU with shared memory and probably L2 cache*/
    computeConvolutionSMCacheGPU<<<grid,block,sharedMemoryI>>>(BdataGPU,BdataOutSMCacheGPU,::RADIUS,nx,ny);
    computeConvolutionSMCacheGPU<<<grid,block,sharedMemoryI>>>(GdataGPU,GdataOutSMCacheGPU,::RADIUS,nx,ny);
    computeConvolutionSMCacheGPU<<<grid,block,sharedMemoryI>>>(RdataGPU,RdataOutSMCacheGPU,::RADIUS,nx,ny);
    CHECK(cudaDeviceSynchronize());
    timerSMCache.stop(false);
    CHECK(cudaGetLastError());
  }
  timerSMCache.averageTime();
  
  /*Copy output data computed using shared memory from GPU to CPU*/
  CHECK(cudaMemcpy(BSMCacheGPU.data(),BdataOutSMCacheGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(GSMCacheGPU.data(),GdataOutSMCacheGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(RSMCacheGPU.data(),RdataOutSMCacheGPU,sizeof(TypeVar)*nx*ny,cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < BGPU.size(); ++i){
    if (BSMIIGPU[i] != BGPU[i] || GSMIIGPU[i] != GGPU[i] || RSMIIGPU[i] != RGPU[i]) {
      std::cerr << "Index: " << i 
    		<< " | B error: " << BSMCacheGPU[i] - BGPU[i] 
  		<< ", G error: " << GSMCacheGPU[i] - GGPU[i] 
   		<< ", R error: " << RSMCacheGPU[i] - RGPU[i] << std::endl;
      break;
    }
  }
  
#ifdef DEBUG
  for(size_t i=0;i<BGPU.size();++i){
    if(std::abs(BConvCPU[i]-BGPU[i])>::epsError || std::abs(GConvCPU[i]-GGPU[i])>::epsError ||
       std::abs(RConvCPU[i]-RGPU[i])>::epsError){
      std::cerr<<"Error : "<<std::abs(BConvCPU[i]-BGPU[i])<<"\t"<<std::abs(GConvCPU[i]-GGPU[i])<<"\t"
	       <<std::abs(RConvCPU[i]-RGPU[i])<<std::endl;
      break;
    }
  }
#endif
  if(::PLOT){
    /* Create output image and assign the processed channels back to OpenCV matrices */
    std::vector<cv::Mat> processedChannels;
    
    processedChannels.push_back(cv::Mat(ny,nx,CV_32F,BSMIGPU.data()));
    processedChannels.push_back(cv::Mat(ny,nx,CV_32F,GSMIGPU.data()));
    processedChannels.push_back(cv::Mat(ny,nx,CV_32F,RSMIGPU.data()));
    
    cv::Mat result;
    cv::merge(processedChannels, result);
    
    /* Convert to 8-bit for display */
    cv::normalize(result,result,0,255,cv::NORM_MINMAX);
    result.convertTo(result,CV_8U);
    
    cv::imshow("Original Image", image);
    cv::imshow("Convolved Image", result);

    /*Save the output image */
    cv::imwrite("convolved_image.png", result);
    
    cv::waitKey(0);
    result.release();
  }
  
  image.release();
  
  CHECK(cudaFree(BdataGPU));
  CHECK(cudaFree(RdataGPU));
  CHECK(cudaFree(GdataGPU));
  CHECK(cudaFree(BdataOutGPU));
  CHECK(cudaFree(RdataOutGPU));
  CHECK(cudaFree(GdataOutGPU));
  CHECK(cudaFree(BdataOutSMIGPU));
  CHECK(cudaFree(RdataOutSMIGPU));
  CHECK(cudaFree(GdataOutSMIGPU));
  CHECK(cudaFree(BdataOutSMIIGPU));
  CHECK(cudaFree(RdataOutSMIIGPU));
  CHECK(cudaFree(GdataOutSMIIGPU));
  CHECK(cudaFree(BdataOutSMCacheGPU));
  CHECK(cudaFree(RdataOutSMCacheGPU));
  CHECK(cudaFree(GdataOutSMCacheGPU));
  
  return EXIT_SUCCESS;
}
