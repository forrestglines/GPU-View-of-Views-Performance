#!/bin/bash

n_var=10
n_buf=2

export OMP_PROC_BIND=spread 
export OMP_PLACES=threads


for dir in "kokkos_zbuffer_pack" "cuda_zbuffer_pack";
do
    echo "" > $dir/timings.dat
    for n_side_log2 in $(seq 3 8);
    do
        n_side=$((2**$n_side_log2))
        n_run=$((2**(22-$n_side_log2)))
        echo "Running $dir $n_side $n_buf $n_run"
        $dir/executable.cuda $n_var $n_side $n_buf $n_run >> $dir/timings.dat
    done
done

for dir in "kokkos_full_domain" "cuda_full_domain";
do
    echo "" > $dir/timings.dat
    for n_grid_log2 in $(seq 10 24);
    do
        n_grid=$((2**$n_grid_log2))
        n_run=$((2**(28-$n_grid_log2)))
        echo "Running $dir $n_grid $n_run"
        $dir/executable.cuda $n_var $n_grid $n_run >> $dir/timings.dat
    done
done
