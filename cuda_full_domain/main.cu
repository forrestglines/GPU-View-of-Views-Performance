
#include <iostream>
#include "cuda.h"

using Real = double;

//Test wrapper to run a function multiple times
template<typename PerfFunc>
float kernel_timer_wrapper(const int n_burn, const int n_perf, PerfFunc perf_func){

  //Initialize the timer and test
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for( int i_run = 0; i_run < n_burn + n_perf; i_run++){

    if(i_run == n_burn){
      //Burn in time is over, start timing
      cudaEventRecord(start);
    }

    //Run the function timing performance
    perf_func();
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  return milliseconds/1000.;
}

__global__ void k_array2d(Real* array2d_in, Real* array2d_out){
  const int i_grid = threadIdx.x + blockDim.x*blockIdx.x;
  const int j_var = blockIdx.y;

  const int n_grid = blockDim.x*gridDim.x;

  array2d_out[j_var*n_grid + i_grid] = 2.*array2d_in[j_var*n_grid + i_grid];
}

__global__ void k_array_of_array1d(Real** array_of_array1d_in, Real** array_of_array1d_out){
  const int i_grid = threadIdx.x + blockDim.x*blockIdx.x;
  const int j_var = blockIdx.y;

  array_of_array1d_out[j_var][i_grid] = 2.*array_of_array1d_in[j_var][i_grid];
}


int main(int argc, char* argv[]) {
  std::size_t pos;
  const int n_var = std::stoi(argv[1],&pos);
  const int n_grid = std::stoi(argv[2],&pos);
  const int n_run = std::stoi(argv[3],&pos);

  const int threads_per_block = 64;
  const dim3 cuda_grid(n_grid/threads_per_block,n_var,1);
  const dim3 cuda_block(threads_per_block,1,1);


  //Setup a raw 2D view
  Real* d_array2d_in;
  cudaMalloc(&d_array2d_in, sizeof(Real)*n_var*n_grid);
  Real* d_array2d_out;
  cudaMalloc(&d_array2d_out, sizeof(Real)*n_var*n_grid);

  float time_array2d = kernel_timer_wrapper( n_run, n_run,
    [&] () {
      k_array2d<<< cuda_grid, cuda_block >>> 
        (d_array2d_in, d_array2d_out);
  });

  //Setup an array of arrays
  //Array of arrays on device
  CUdeviceptr* d_array_of_array1d_in;
  cudaMalloc(&d_array_of_array1d_in, sizeof(CUdeviceptr)*n_var);
  CUdeviceptr* d_array_of_array1d_out;
  cudaMalloc(&d_array_of_array1d_out, sizeof(CUdeviceptr)*n_var);

  //Array of arrays on host
  CUdeviceptr* h_array_of_array1d_in  = (CUdeviceptr*) malloc(sizeof(CUdeviceptr)*n_var);
  CUdeviceptr* h_array_of_array1d_out = (CUdeviceptr*) malloc(sizeof(CUdeviceptr)*n_var);

  //Malloc each 1d array
  for(int i = 0; i < n_var; i++) {
        cudaMalloc((void**)(h_array_of_array1d_in+i ), n_grid * sizeof(Real));
        cudaMalloc((void**)(h_array_of_array1d_out+i), n_grid * sizeof(Real));
  }

  //Move h_array_of_array1d to d_array_of_array1d
  cudaMemcpy(d_array_of_array1d_in, h_array_of_array1d_in, sizeof(CUdeviceptr) * n_var, cudaMemcpyHostToDevice);
  cudaMemcpy(d_array_of_array1d_out, h_array_of_array1d_out, sizeof(CUdeviceptr) * n_var, cudaMemcpyHostToDevice);


  double time_array_of_array1d = kernel_timer_wrapper( n_run, n_run,
    [&] () {
      k_array_of_array1d<<< cuda_grid, cuda_block >>> 
        ( (Real**) d_array_of_array1d_in, (Real**) d_array_of_array1d_out);
  });


  double cell_cycles_per_second_array2d = static_cast<double>(n_grid)*static_cast<double>(n_run)/time_array2d; 
  double cell_cycles_per_second_array_of_array1d = static_cast<double>(n_grid)*static_cast<double>(n_run)/time_array_of_array1d; 
  std::cout<< n_var << " " << n_grid << " " << n_run << " " << time_array2d << " " << time_array_of_array1d << " " 
           << cell_cycles_per_second_array2d << " " << cell_cycles_per_second_array_of_array1d << std::endl;


  //free each 1d array
  for(int i = 0; i < n_var; i++) {
        cudaFree(h_array_of_array1d_in+i);
        cudaFree(h_array_of_array1d_out+i);
  }

  free(h_array_of_array1d_in);
  free(h_array_of_array1d_out);
  cudaFree(d_array_of_array1d_in);
  cudaFree(d_array_of_array1d_out);

  cudaFree(d_array2d_in);
  cudaFree(d_array2d_out);


}
