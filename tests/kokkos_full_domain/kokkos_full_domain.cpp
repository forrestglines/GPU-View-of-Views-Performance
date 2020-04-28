#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

using Real = double;
using View1D =  Kokkos::View<Real*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using View2D =  Kokkos::View<Real**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using ViewOfView1D =  Kokkos::View<View1D* >;

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
    const int n_grid = std::stoi(argv[2],&pos);
    const int n_run = std::stoi(argv[3],&pos);

    auto policy = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0,n_var*n_grid,Kokkos::ChunkSize(512));

    //Setup a raw 2D view
    View2D view2d_in("view2D_in",n_grid,n_var);
    View2D view2d_out("view2D_out",n_grid,n_var);

    double time_view2d = kernel_timer_wrapper( n_run, n_run,
      [&] () {
        Kokkos::parallel_for( "View2D Loop", policy,
          KOKKOS_LAMBDA (const int &i){
            const int i_grid = i%n_grid;
            const int j_var = i/n_grid;

            view2d_out(i_grid,j_var) = 2.*view2d_in(i_grid,j_var);

        }); 
    });


    //Setup a view of views
    ViewOfView1D  view_of_view1d_in("view_of_view1d_in",n_var);
    ViewOfView1D  view_of_view1d_out("view_of_view1d_out",n_var);
    auto h_view_of_view1d_in = Kokkos::create_mirror_view(view_of_view1d_in);
    auto h_view_of_view1d_out = Kokkos::create_mirror_view(view_of_view1d_out);
    for( int i=0; i < n_var; i++){
      h_view_of_view1d_in[i] = View1D("view1d_in",n_grid);
      h_view_of_view1d_out[i] = View1D("view1d_out",n_grid);
    }
    Kokkos::deep_copy(view_of_view1d_in,h_view_of_view1d_in);
    Kokkos::deep_copy(view_of_view1d_out,h_view_of_view1d_out);

    double time_view_of_view1d = kernel_timer_wrapper( n_run, n_run,
      [&] () {
        Kokkos::parallel_for( policy,
          KOKKOS_LAMBDA (const int &i){
            const int i_grid = i%n_grid;
            const int j_var = i/n_grid;

            //Get the 1D views
            auto view1d_in = view_of_view1d_in(j_var);
            auto view1d_out = view_of_view1d_out(j_var);

            view1d_out(i_grid) = 2.*view1d_in(i_grid);

        }); 
    });

    double cell_cycles_per_second_view2d = static_cast<double>(n_grid)*static_cast<double>(n_run)/time_view2d; 
    double cell_cycles_per_second_view_of_view1d = static_cast<double>(n_grid)*static_cast<double>(n_run)/time_view_of_view1d; 
    std::cout<< n_var << " " << n_grid << " " << n_run  << " " << time_view2d << " " << time_view_of_view1d << " " 
             << cell_cycles_per_second_view2d << " " << cell_cycles_per_second_view_of_view1d << std::endl;
  }
  Kokkos::finalize();
}
