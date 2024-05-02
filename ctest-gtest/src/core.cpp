// Main solver routines for heat equation solver

#include <mpi.h>

#include "field.hpp"
#include "parallel.hpp"

// Exchange the boundary values
void exchange(Field &field, const ParallelData &parallel) {

    // Send to up, receive from down
    auto sbuf = field.data(1, 0);
    auto rbuf = field.data(field.num_rows + 1, 0);
    MPI_Sendrecv(sbuf, field.num_cols + 2, MPI_DOUBLE, parallel.nup, 11, rbuf,
                 field.num_cols + 2, MPI_DOUBLE, parallel.ndown, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    sbuf = field.data(field.num_rows, 0);
    rbuf = field.data();
    MPI_Sendrecv(sbuf, field.num_cols + 2, MPI_DOUBLE, parallel.ndown, 12, rbuf,
                 field.num_cols + 2, MPI_DOUBLE, parallel.nup, 12,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Update the temperature values using five-point stencil */
void evolve(Field &curr, const Field &prev, double diffusion_constant,
            double dt) {
    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    for (int i = 0; i < curr.num_rows; i++) {
        for (int j = 0; j < curr.num_cols; j++) {
            curr(i, j) =
                prev(i, j) +
                diffusion_constant * dt *
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
