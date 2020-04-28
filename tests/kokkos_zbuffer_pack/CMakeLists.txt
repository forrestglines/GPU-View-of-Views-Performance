add_executable(
    kokkos_zbuffer_pack
        kokkos_zbuffer_pack.cpp
)
target_link_libraries(kokkos_zbuffer_pack PUBLIC Kokkos::kokkos)
