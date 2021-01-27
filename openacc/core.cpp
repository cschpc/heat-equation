/* Main solver routines for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"

/* Exchange the boundary values */
void exchange(field *temperature, parallel_data *parallel)
{
    double *data;  
    double *sbuf_up, *sbuf_down, *rbuf_up, *rbuf_down;

    data = temperature->data;

    // Send to the up, receive from down
#pragma acc host_data use_device(data)
    {
      sbuf_up = data + temperature->ny + 2; // upper data
      rbuf_down = data + (temperature->nx + 1) * (temperature->ny + 2); // lower halo
    }
    MPI_Sendrecv(sbuf_up, temperature->ny + 2, MPI_DOUBLE,
                 parallel->nup, 11,
                 rbuf_down, temperature->ny + 2, MPI_DOUBLE, 
                 parallel->ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to the down, receive from up
#pragma acc host_data use_device(data)
    {
      sbuf_down = data + temperature->nx * (temperature->ny + 2); // lower data
      rbuf_up = data; // upper halo
    }
    MPI_Sendrecv(sbuf_down, temperature->ny + 2, MPI_DOUBLE, 
                 parallel->ndown, 12,
                 rbuf_up, temperature->ny + 2, MPI_DOUBLE,
                 parallel->nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}


/* Update the temperature values using five-point stencil */
void evolve(field *curr, field *prev, double a, double dt)
{
  double dx2, dy2;
  int nx, ny;
  double *currdata, *prevdata;

  /* HINT: to help the compiler do not access members of structures
   * within OpenACC parallel regions */
  currdata = curr->data;
  prevdata = prev->data;
  nx = curr->nx;
  ny = curr->ny;

  /* Determine the temperature field at next time step
   * As we have fixed boundary conditions, the outermost gridpoints
   * are not updated. */
  dx2 = prev->dx * prev->dx;
  dy2 = prev->dy * prev->dy;
#pragma acc parallel loop \
    present(prevdata[0:(nx+2)*(ny+2)], currdata[0:(nx+2)*(ny+2)]) collapse(2)
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

/* Start a data region and copy temperature fields to the device */
void enter_data(field *curr, field *prev)
{
    int nx, ny;
    double *currdata, *prevdata;

    currdata = curr->data;
    prevdata = prev->data;
    nx = curr->nx;
    ny = curr->ny;

#pragma acc enter data \
    copyin(currdata[0:(nx+2)*(ny+2)], prevdata[0:(nx+2)*(ny+2)])
}

/* End a data region and copy temperature fields back to the host */
void exit_data(field *curr, field *prev)
{
    int nx, ny;
    double *currdata, *prevdata;

    currdata = curr->data;
    prevdata = prev->data;
    nx = curr->nx;
    ny = curr->ny;

#pragma acc exit data \
    copyout(currdata[0:(nx+2)*(ny+2)], prevdata[0:(nx+2)*(ny+2)])
}

/* Copy a temperature field from the device to the host */
void update_host(field *temperature)
{
    int nx, ny;
    double *data;

    data = temperature->data;
    nx = temperature->nx;
    ny = temperature->ny;

#pragma acc update host(data[0:(nx+2)*(ny+2)])
}

/* Copy a temperature field from the host to the device */
void update_device(field *temperature)
{
    int nx, ny;
    double *data;

    data = temperature->data;
    nx = temperature->nx;
    ny = temperature->ny;

#pragma acc update device(data[0:(nx+2)*(ny+2)])
}
