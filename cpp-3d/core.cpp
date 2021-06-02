// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values
void exchange(Field& field, const ParallelData parallel)
{

    size_t buf_size = (field.ny + 2) * (field.nz + 2);
    // Send to up, receive from down
    auto sbuf = field.temperature.data(1, 0, 0);
    auto rbuf  = field.temperature.data(field.nx + 1, 0, 0);
    MPI_Sendrecv(sbuf, buf_size, MPI_DOUBLE,
                 parallel.nup, 11,
                 rbuf, buf_size, MPI_DOUBLE, 
                 parallel.ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    sbuf = field.temperature.data(field.nx, 0, 0);
    rbuf = field.temperature.data();
    MPI_Sendrecv(sbuf, buf_size, MPI_DOUBLE, 
                 parallel.ndown, 12,
                 rbuf, buf_size, MPI_DOUBLE,
                 parallel.nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);

  auto dx2 = (prev.dx * prev.dx);
  auto dy2 = (prev.dy * prev.dy);
  auto dz2 = (prev.dz * prev.dz);
  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 1; i < curr.nx + 1; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
#pragma omp simd
      for (int k = 1; k < curr.nz + 1; k++) {
            curr(i, j, k) = prev(i, j, k) + a * dt * (
	       //( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       // ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2 +
	       //  ( prev(i, j, k + 1) - 2.0 * prev(i, j, k) + prev(i, j, k - 1) ) * inv_dz2
	       // ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) / (prev.dx*prev.dx) +
	       // ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) / (prev.dy*prev.dy)
	        ( prev(i + 1, j, k) - 2.0 * prev(i, j, k) + prev(i - 1, j, k) ) / (dx2) +
	        ( prev(i, j + 1, k) - 2.0 * prev(i, j, k) + prev(i, j - 1, k) ) / (dy2) +
	        ( prev(i, j, k + 1) - 2.0 * prev(i, j, k) + prev(i, j, k - 1) ) / (dz2) 
               );
      }
    }
  }

}
