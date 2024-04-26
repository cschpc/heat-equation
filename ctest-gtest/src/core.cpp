// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values
void exchange(Field &field, const ParallelData &parallel) {

    // Send to up, receive from down
    auto sbuf = field.temperature.data(1, 0);
    auto rbuf = field.temperature.data(field.num_rows + 1, 0);
    MPI_Sendrecv(sbuf, field.num_cols + 2, MPI_DOUBLE, parallel.nup, 11, rbuf,
                 field.num_cols + 2, MPI_DOUBLE, parallel.ndown, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    sbuf = field.temperature.data(field.num_rows, 0);
    rbuf = field.temperature.data();
    MPI_Sendrecv(sbuf, field.num_cols + 2, MPI_DOUBLE, parallel.ndown, 12, rbuf,
                 field.num_cols + 2, MPI_DOUBLE, parallel.nup, 12,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 1; i < curr.num_rows + 1; i++) {
      for (int j = 1; j < curr.num_cols + 1; j++) {
          curr(i, j) =
              prev(i, j) +
              a * dt *
                  ((prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j)) *
                       Field::inv_dx2 +
                   (prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1)) *
                       Field::inv_dy2);
      }
  }
}

namespace heat {
double stencil(int i, int j, const Field &field, double a, double dt) {
    return 0.0;
}
} // namespace heat
