// Main solver routines for heat equation solver

#include <mpi.h>

#include "constants.hpp"
#include "core.hpp"
#include "field.hpp"
#include "parallel.hpp"

namespace heat {
// Exchange the boundary values
void exchange(Field &field, const ParallelData &parallel) {

    // Send to up, receive from down
    constexpr int tag1 = 11;
    MPI_Sendrecv(field.to_up(), field.num_cols, MPI_DOUBLE, parallel.nup, tag1,
                 field.from_down(), field.num_cols, MPI_DOUBLE, parallel.ndown,
                 tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    constexpr int tag2 = 12;
    MPI_Sendrecv(field.to_down(), field.num_cols, MPI_DOUBLE, parallel.ndown,
                 tag2, field.from_up(), field.num_cols, MPI_DOUBLE,
                 parallel.nup, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Update the temperature values using five-point stencil */
void evolve(Field &curr, const Field &prev, const heat::Constants &constants) {
    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    for (int i = 0; i < curr.num_rows; i++) {
        for (int j = 0; j < curr.num_cols; j++) {
            curr(i, j) =
                prev(i, j) +
                constants.a * constants.dt *
                    ((prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1)) *
                         constants.inv_dx2 +
                     (prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j)) *
                         constants.inv_dy2);
        }
    }
}
} // namespace heat
