add_executable(
    kokkos_full_domain
        kokkos_full_domain.cpp
)
target_link_libraries(kokkos_full_domain PUBLIC Kokkos::kokkos)
