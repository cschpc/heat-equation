/* Main solver routines for heat equation solver */

#include <mpi.h>

#include "heat.hpp"

/* Exchange the boundary values */
void exchange(Field& field , ParallelData const parallel)
{
    double *data;  
    double *sbuf_up, *sbuf_down, *rbuf_up, *rbuf_down;

    data = field.temperature.data();

    // Send to the up, receive from down
    sbuf_up = data + field.ny + 2; // upper data
    rbuf_down = data + (field.nx + 1) * (field.ny + 2); // lower halo

    MPI_Sendrecv(sbuf_up, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 11,
                 rbuf_down, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to the down, receive from up
    sbuf_down = data + field.nx * (field.ny + 2); // lower data
    rbuf_up = data; // upper halo

    MPI_Sendrecv(sbuf_down, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 12,
                 rbuf_up, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

/* Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, double const a, double const dt)
{

  /* HINT: to help the compiler do not access members of structures
   * within OpenACC parallel regions */
  auto currdata = curr.temperature.data();
  auto prevdata = prev.temperature.data();
  auto nx = curr.nx;
  auto ny = curr.ny;

  /* Determine the temperature field at next time step
   * As we have fixed boundary conditions, the outermost gridpoints
   * are not updated. */
  auto dx2 = prev.dx * prev.dx;
  auto dy2 = prev.dy * prev.dy;
  for (int i = 1; i < nx + 1; i++) {
    for (int j = 1; j < ny + 1; j++) {
            int ind = i * (ny + 2) + j;
            int ip = (i + 1) * (ny + 2) + j;
            int im = (i - 1) * (ny + 2) + j;
	    int jp = i * (ny + 2) + j + 1;
	    int jm = i * (ny + 2) + j - 1;
            currdata[ind] = prevdata[ind] + a * dt *
	      ((prevdata[ip] -2.0 * prevdata[ind] + prevdata[im]) / dx2 +
	       (prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm]) / dy2);
    }
  }

}
