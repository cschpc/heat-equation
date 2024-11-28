// Utility functions for heat equation solver
//    NOTE: This file does not need to be edited! 

#ifdef NO_MPI
#include <omp.h>
#else
#include <mpi.h>
#endif
#include "heat.hpp"
#include <Kokkos_Core.hpp>

// Calculate average temperature
double average(const Field& field)
{
     double local_average = 0.0;
     double average = 0.0;

     using MDPolicyType3D = Kokkos::MDRangePolicy<Kokkos::Rank<3> >;
     MDPolicyType3D mdpolicy_3d({1, 1, 1}, {field.nx + 1, field.ny + 1, field.nz + 1});

     Kokkos::parallel_reduce("average", mdpolicy_3d,
     KOKKOS_LAMBDA(const int i, const int j, const int k, double& local_average) {
           local_average += field.temperature(i, j, k);
         }, local_average);

#ifdef NO_MPI
     average = local_average;
#else
     MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
     average /= (field.nx_full * field.ny_full * field.nz_full);
     return average;
}

double timer() 
{
    double t0;
#ifdef NO_MPI
    t0 = omp_get_wtime();
#else
    t0 = MPI_Wtime();
#endif
    return t0;
}

