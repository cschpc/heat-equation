// Utility functions for heat equation solver
//    NOTE: This file does not need to be edited! 

#include <mpi.h>

#include "heat.hpp"

// Calculate average temperature
double average(const Field& field)
{
     double local_average = 0.0;
     double average = 0.0;

     for (int i = 1; i < field.nx + 1; i++) {
       for (int j = 1; j < field.ny + 1; j++) {
         for (int k = 1; k < field.nz + 1; k++) {
           local_average += field.temperature(i, j, k);
         }
       }
     }

     MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     average /= (field.nx_full * field.ny_full * field.nz_full);
     return average;
}
