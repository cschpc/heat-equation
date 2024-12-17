// Utility functions for heat equation solver
// NOTE: This file does not need to be edited! 

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "heat.hpp"

// Calculate average temperature
double average(const Field& field)
{
     double local_average = 0.0;
     double average = 0.0;

    using MDPolicyType2D = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
    MDPolicyType2D mdpolicy_2d({1, 1}, {field.nx + 1, field.ny + 1});

    Kokkos::parallel_reduce("average", mdpolicy_2d, 
      KOKKOS_LAMBDA(const int i, const int j, double& local_average) {
         local_average += field.temperature(i, j);
       }, local_average);

     MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     average /= (field.nx_full * field.ny_full);
     return average;
}
