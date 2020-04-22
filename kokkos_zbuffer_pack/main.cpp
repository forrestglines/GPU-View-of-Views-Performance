#include "Kokkos_Core.hpp"

#include <iostream>
#include <vector>

using Real = double;
using View1D =  Kokkos::View<Real*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using View3D =  Kokkos::View<Real***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using View4D =  Kokkos::View<Real****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using ViewOfView3D =  Kokkos::View<View3D* >;

//Test wrapper to run a function multiple times
template<typename PerfFunc>
double kernel_timer_wrapper(const int n_burn, const int n_perf, PerfFunc perf_func){

  //Initialize the timer and test
  Kokkos::Timer timer;

  for( int i_run = 0; i_run < n_burn + n_perf; i_run++){

    if(i_run == n_burn){
      //Burn in time is over, start timing
      Kokkos::fence();
      timer.reset();
    }

    //Run the function timing performance
    perf_func();
  }

  //Time it
  Kokkos::fence();
  double perf_time = timer.seconds();

  return perf_time;

}


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::size_t pos;
    const int n_var = std::stoi(argv[1],&pos);
    const int n_side = std::stoi(argv[2],&pos);
    const int n_buf = std::stoi(argv[3],&pos);
    const int n_run = std::stoi(argv[4],&pos);

    //Order of iteration, fastest moving to slowest moving:
    // x (full n_side), y (full n_side), z (only n_buf), var (n_var)
    const int buf_size = n_side*n_side*n_buf*n_var;
    const int n_side2n_buf = n_side*n_side*n_buf;
    const int n_side2 = n_side*n_side;

    auto policy = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0,buf_size,Kokkos::ChunkSize(512));
    
    //Setup a 1D buffer
    View1D view1d_buf("view1d_buf",buf_size);

    //Setup a raw 4D view
    View4D view4d_in ("view4D_in", n_side,n_side,n_side,n_var);

    //Test packing the z buffer
    double time_view4d = kernel_timer_wrapper( n_run, n_run,
      [&] () {
        Kokkos::parallel_for( "View4D buffer_pack", policy,
          KOKKOS_LAMBDA (const int &idx){

            const int v_var = idx / n_side2n_buf;
            const int k_grid = (idx - v_var * n_side2n_buf) / n_side2;
            const int j_grid = (idx - v_var * n_side2n_buf - k_grid * n_side2) / n_side;
            const int i_grid = idx - v_var * n_side2n_buf - k_grid * n_side2 - j_grid * n_side;

            view1d_buf(idx) = view4d_in(i_grid,j_grid,k_grid+n_buf,v_var);

        }); 
    });


    //Setup a view of views
    ViewOfView3D  view_of_view3d_in("view_of_view3d_in",n_var);
    auto h_view_of_view3d_in = Kokkos::create_mirror_view(view_of_view3d_in);
    for( int i=0; i < n_var; i++){
      h_view_of_view3d_in[i] = View3D("view3d_in",n_side,n_side,n_side);
    }
    Kokkos::deep_copy(view_of_view3d_in,h_view_of_view3d_in);

    //Test packing the z buffer
    double time_view_of_view3d = kernel_timer_wrapper( n_run, n_run,
      [&] () {
        Kokkos::parallel_for( policy,
          KOKKOS_LAMBDA (const int &idx){
            const int v_var = idx / n_side2n_buf;
            const int k_grid = (idx - v_var * n_side2n_buf) / n_side2;
            const int j_grid = (idx - v_var * n_side2n_buf - k_grid * n_side2) / n_side;
            const int i_grid = idx - v_var * n_side2n_buf - k_grid * n_side2 - j_grid * n_side;

            //Get the 3D views
            auto view3d_in = view_of_view3d_in(v_var);

            view1d_buf(idx) = view3d_in(i_grid,j_grid,k_grid+n_buf);

        }); 
    });

    double cell_cycles_per_second_view4d = static_cast<double>(n_side2n_buf)*static_cast<double>(n_run)/time_view4d; 
    double cell_cycles_per_second_view_of_view3d = static_cast<double>(n_side2n_buf)*static_cast<double>(n_run)/time_view_of_view3d; 
    std::cout<< n_var << " " << n_side << " " << n_run << " " << n_buf << " " << time_view4d << " " << time_view_of_view3d << " " 
             << cell_cycles_per_second_view4d << " " << cell_cycles_per_second_view_of_view3d << std::endl;
  }
  Kokkos::finalize();
}
