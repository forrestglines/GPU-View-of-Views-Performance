
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

//A few integers that are needed for indexing on device
__constant__ int c_n_side3;
__constant__ int c_n_side2;
__constant__ int c_n_side;
__constant__ int c_n_side2n_buf;
__constant__ int c_n_buf;

__global__ void k_zbuffer_pack_array4d(Real* array4d_in, Real* array1d_buf){

  const int idx =  threadIdx.x + blockIdx.x*blockDim.x;

  const int v_var = idx / c_n_side2n_buf;
  const int k_grid = (idx - v_var * c_n_side2n_buf) / c_n_side2;
  const int j_grid = (idx - v_var * c_n_side2n_buf - k_grid * c_n_side2) / c_n_side;
  const int i_grid = idx - v_var * c_n_side2n_buf - k_grid * c_n_side2 - j_grid * c_n_side;

  array1d_buf[idx] = array4d_in[i_grid + j_grid*c_n_side + (k_grid+c_n_buf)*c_n_side2 + v_var*c_n_side3];
}

__global__ void k_zbuffer_pack_array_of_array3d(Real** array_of_array3d_in, Real* array1d_buf){
  const int idx =  threadIdx.x + blockIdx.x*blockDim.x;

  const int v_var = idx / c_n_side2n_buf;
  const int k_grid = (idx - v_var * c_n_side2n_buf) / c_n_side2;
  const int j_grid = (idx - v_var * c_n_side2n_buf - k_grid * c_n_side2) / c_n_side;
  const int i_grid = idx - v_var * c_n_side2n_buf - k_grid * c_n_side2 - j_grid * c_n_side;

  array1d_buf[idx] = array_of_array3d_in[v_var][i_grid + j_grid*c_n_side + (k_grid+c_n_buf)*c_n_side2];
}


int main(int argc, char* argv[]) {
  std::size_t pos;
  const int n_var = std::stoi(argv[1],&pos);
  const int n_side = std::stoi(argv[2],&pos);
  const int n_buf = std::stoi(argv[3],&pos);
  const int n_run = std::stoi(argv[4],&pos);

  //Order of iteration, fastest moving to slowest moving:
  // x (full n_side), y (full n_side), z (only n_buf), var (n_var)
  const int buf_size = n_side*n_side*n_buf*n_var;
  const int n_side3 = n_side*n_side*n_side;
  const int n_side2 = n_side*n_side;
  const int n_side2n_buf = n_side*n_side*n_buf;

  const int n_grid = n_side3;

  const int threads_per_block = 128;
  const int cuda_grid = buf_size/threads_per_block;
  const int cuda_block = threads_per_block;


  //Move useful indexing variables into constant memory
  cudaMemcpyToSymbol(c_n_side3, &n_side3, sizeof(n_side3));
  cudaMemcpyToSymbol(c_n_side2, &n_side2, sizeof(n_side2));
  cudaMemcpyToSymbol(c_n_side, &n_side, sizeof(n_side));
  cudaMemcpyToSymbol(c_n_side2n_buf, &n_side2n_buf, sizeof(n_side2n_buf));
  cudaMemcpyToSymbol(c_n_buf, &n_buf, sizeof(n_buf));

  //Setup a 1d view for the buffer
  Real* d_array1d_buf;
  cudaMalloc(&d_array1d_buf, sizeof(Real)*buf_size);

  //Setup a raw 4D view
  Real* d_array4d_in;
  cudaMalloc(&d_array4d_in, sizeof(Real)*n_var*n_grid);

  float time_array4d = kernel_timer_wrapper( n_run, n_run,
    [&] () {
      k_zbuffer_pack_array4d<<< cuda_grid, cuda_block >>> 
        (d_array4d_in, d_array1d_buf);
  });

  //Setup an array of arrays
  //Array of arrays on device
  CUdeviceptr* d_array_of_array3d_in;
  cudaMalloc(&d_array_of_array3d_in, sizeof(CUdeviceptr)*n_var);

  //Array of arrays on host
  CUdeviceptr* h_array_of_array3d_in  = (CUdeviceptr*) malloc(sizeof(CUdeviceptr)*n_var);

  //Malloc each 3d array
  for(int i = 0; i < n_var; i++) {
        cudaMalloc((void**)(h_array_of_array3d_in+i ), n_grid * sizeof(Real));
  }

  //Move h_array_of_array1d to d_array_of_array3d
  cudaMemcpy(d_array_of_array3d_in, h_array_of_array3d_in, sizeof(CUdeviceptr) * n_var, cudaMemcpyHostToDevice);


  double time_array_of_array3d = kernel_timer_wrapper( n_run, n_run,
    [&] () {
      k_zbuffer_pack_array_of_array3d<<< cuda_grid, cuda_block >>> 
        ( (Real**) d_array_of_array3d_in, d_array1d_buf);
  });


  double cell_cycles_per_second_array4d = static_cast<double>(n_side2n_buf)*static_cast<double>(n_run)/time_array4d; 
  double cell_cycles_per_second_array_of_array3d = static_cast<double>(n_side2n_buf)*static_cast<double>(n_run)/time_array_of_array3d; 
  std::cout<< n_var << " " << n_side << " " << n_run << " " << n_buf << " " << time_array4d << " " << time_array_of_array3d << " " 
           << cell_cycles_per_second_array4d << " " << cell_cycles_per_second_array_of_array3d << std::endl;


  //free each 1d array
  for(int i = 0; i < n_var; i++) {
        cudaFree(h_array_of_array3d_in+i);
  }

  free(h_array_of_array3d_in);
  cudaFree(d_array_of_array3d_in);

  cudaFree(d_array4d_in);

  cudaFree(d_array1d_buf);


}
