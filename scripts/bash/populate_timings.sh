#!/bin/bash

n_var=9
n_buf=2
n_vector=3

export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

for name in "parthenon_full_domain" ;
do
    echo "" > $name-timings.dat
    for n_side_log2 in $(seq 3 8);
    do
        n_side=$((2**$n_side_log2))
        n_run=$((2**(20-2*$n_side_log2)))
        echo "Running $name $n_var $n_vector $n_side $n_run"
        tests/$name/$name $n_var $n_vector $n_side $n_run >> $name-timings.dat
    done
done

for name in "kokkos_zbuffer_pack" "cuda_zbuffer_pack";
do
    echo "" > $name-timings.dat
    for n_side_log2 in $(seq 3 8);
    do
        n_side=$((2**$n_side_log2))
        n_run=$((2**(22-$n_side_log2)))
        echo "Running $name $n_side $n_buf $n_run"
        tests/$name/$name $n_var $n_side $n_buf $n_run >> $name-timings.dat
    done
done

for name in "kokkos_full_domain" "cuda_full_domain";
do
    echo "" > $name-timings.dat
    for n_grid_log2 in $(seq 10 24);
    do
        n_grid=$((2**$n_grid_log2))
        n_run=$((2**(28-$n_grid_log2)))
        echo "Running $name $n_grid $n_vector $n_run"
        tests/$name/$name $n_var $n_grid $n_run >> $name-timings.dat
    done
done

