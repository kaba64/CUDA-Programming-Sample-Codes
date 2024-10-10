#pragma once


#define CHECK(argu) {                                          \
  const cudaError_t error = argu;                              \
  if (error != cudaSuccess) {                                  \
    std::cerr << "Error: " << __FILE__ << ":" << __LINE__      \
              << ", code: " << error                           \
              << ", reason: " << cudaGetErrorString(error)     \
              << std::endl;                                    \
    exit(1);                                                   \
  }                                                            \
}

class Timer {
  std::string kernelName;
  std::vector<float> timeSpentAver;
  float timeSpent;
  const size_t Id;
  cudaStream_t stream;
  cudaEvent_t startEvent, endEvent;
public:
  Timer(std::string kernelNameIn,size_t IdIn=0,cudaStream_t streamIn=0) :
    kernelName{kernelNameIn}, Id{IdIn}, stream{streamIn},timeSpent{0.0f}{
    CHECK(cudaSetDevice(Id));
    CHECK(cudaEventCreate(&startEvent));
    CHECK(cudaEventCreate(&endEvent));
  }
  void start ( ){
    CHECK(cudaSetDevice(Id));
    CHECK(cudaEventRecord(startEvent,stream));
  }
  void stop (bool printTime = true) {
    CHECK(cudaSetDevice(Id));
    CHECK(cudaEventRecord(endEvent,stream));
    CHECK(cudaEventSynchronize(endEvent));
    CHECK(cudaEventElapsedTime(&timeSpent,startEvent,endEvent));
    timeSpentAver.push_back(timeSpent);
    if (printTime) {
      std::cout << "Time spent at kernel " << kernelName 
                << " in iteration " << timeSpentAver.size() 
                << " is " << timeSpent << " (ms)" << std::endl;
    }
  }
  void averageTime(){
    if (!timeSpentAver.empty()) {
      float sum = std::accumulate(timeSpentAver.begin(), timeSpentAver.end(),0.0f);
      std::cout << "The average time spent at kernel " << kernelName 
                << " is " << sum / timeSpentAver.size() << " (ms)" << std::endl;
    } else {
      std::cout << "No time measurements available for kernel " << kernelName << std::endl;
    }
  }
  ~Timer ( ) {
    CHECK(cudaSetDevice(Id));
    CHECK(cudaEventDestroy(startEvent));
    CHECK(cudaEventDestroy(endEvent));
  }
};

template<typename T>
class Coefficient{
public:
  Coefficient(T al_in,T lx_in,T ly_in, T lz_in,size_t nx_in,size_t ny_in, size_t nz_in,
	      T dt_in,T time_in,T bx_in,T tx_in,T by_in,T ty_in,T bz_in,T tz_in):
    alpha{al_in}, l{lx_in,ly_in,lz_in},nc{nx_in,ny_in,nz_in},dt{dt_in}, time{time_in},
    bc{{bx_in, tx_in}, {by_in, ty_in}, {bz_in, tz_in}}{
    dl[0]   = l[0]/nc[0];
    dl[1]   = l[1]/nc[1];
    dl[2]   = l[2]/nc[2];
    nStep   = static_cast<size_t>(std::round(time/dt));    
  }
  
  void set_dt(const T dt_in){dt=dt_in;}
  
  T alpha;               // Thermal diffusivity
  T l[3], dl[3];         // l[3] : length in x, y, and z directions
  size_t nc[3]; // Number of grid points in x and y directions
  T time,dt;
  size_t nStep;
  T bc[3][2];           // <0 : x, 1 : y, 2 : z > bounday conditions
};

template<typename T>
void apply_boundary_conditions(std::vector<T>& u,const Coefficient<T> &coe) {
  
  size_t n[3] = {coe.nc[0],coe.nc[1],coe.nc[2]};
  
  /*z-y plane x = 0 and x = 1*/
  for (size_t k = 0; k < n[2]; ++k) {
    for (size_t j = 0; j < n[1]; ++j){
      u[n[0]*n[1]*k+n[1]*j+0]         = coe.bc[0][0];
      u[n[0]*n[1]*k+n[1]*j+(n[0]-1)]  = coe.bc[0][1];
    }
  }
  /*x-z plane y = 0 and y = 1*/
  for (size_t k = 0; k < n[2]; ++k) {
    for (size_t i = 0; i < n[0]; ++i){
      u[n[0]*n[1]*k+n[1]*0+i]         = coe.bc[1][0];
      u[n[0]*n[1]*k+n[1]*(n[1]-1)+i]  = coe.bc[1][1];
    }
  }
  /*x-y plane z = 0  and z = 1*/
  for (size_t j = 0; j < n[1]; ++j) {
    for (size_t i = 0; i < n[0]; ++i){
      u[n[0]*n[1]*0+n[1]*j+i]         = coe.bc[2][0];
      u[n[0]*n[1]*(n[2]-1)+n[1]*j+i]  = coe.bc[2][1];
    }
  }
}

template<typename T1,typename T2>
void setCoefficientEquation(std::vector<T1>& v,const Coefficient<T2>& coe){
  for(size_t i=0;i<v.size();++i){
    size_t id = i<3 ? 0 : (i<5 ? 1 : 2);
    if(i==0){
      v[i] = (-2.0*coe.alpha*coe.dt)/(coe.dl[0]*coe.dl[0])+
	     (-2.0*coe.alpha*coe.dt)/(coe.dl[1]*coe.dl[1])+
	     (-2.0*coe.alpha*coe.dt)/(coe.dl[2]*coe.dl[2])+1.0;
    }else{
      v[i] = (coe.alpha*coe.dt)/(coe.dl[id]*coe.dl[id]);
    }
  }
}

template<typename T>
void write_to_file(const std::vector<T>& u, const Coefficient<T>& coe, size_t timestep) {
  std::ofstream file;
  file.open("output_" + std::to_string(timestep) + ".csv");
  
  for (size_t k = 0; k < coe.nc[2]; ++k) {
    for (size_t j = 0; j < coe.nc[1]; ++j) {
      for (size_t i = 0; i < coe.nc[0]; ++i) {
        file << u[coe.nc[0]*coe.nc[1]*k+coe.nc[1]*j+i];
        if (i < coe.nc[0] - 1)
	  file << ",";
      }
      file << "\n";
    }
    file << "\n";
  }
  file.close();
}