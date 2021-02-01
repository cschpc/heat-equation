#include "hip/hip_runtime.h"
/* Main solver routines for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <hip/hip_runtime_api.h>

#include "heat.h"

/* Update the temperature values using five-point stencil */
__global__ void evolve_kernel(double *currdata, double *prevdata, double a, double dt, int nx, int ny,
                       double dx2, double dy2)
{

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    int ind, ip, im, jp, jm;

    // CUDA threads are arranged in column major order; thus j index from x, i from y
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;        

    if (i > 0 && j > 0 && i < nx+1 && j < ny+1) {
        ind = i * (ny + 2) + j;
        ip = (i + 1) * (ny + 2) + j;
        im = (i - 1) * (ny + 2) + j;
	jp = i * (ny + 2) + j + 1;
	jm = i * (ny + 2) + j - 1;
        currdata[ind] = prevdata[ind] + a * dt *
	      ((prevdata[ip] -2.0 * prevdata[ind] + prevdata[im]) / dx2 +
	      (prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm]) / dy2);

    }

}

void evolve(field *curr, field *prev, double a, double dt)
{
    int nx, ny;
    double dx2, dy2;
    nx = prev->nx;
    ny = prev->ny;
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;

    /* CUDA thread settings */
    const int blocksize = 16;  //!< CUDA thread block dimension
    dim3 dimBlock(blocksize, blocksize); 
    dim3 dimGrid((nx + 2 + blocksize - 1) / blocksize, 
                 (ny + 2 + blocksize - 1) / blocksize); 

    hipLaunchKernelGGL(evolve_kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, curr->devdata, prev->devdata, a, dt, nx, ny, dx2, dy2);
    hipDeviceSynchronize();
}

void enter_data(field *temperature1, field *temperature2)
{
    size_t datasize;

    datasize = (temperature1->nx + 2) * (temperature1->ny + 2) * sizeof(double);
  
    hipMalloc(&temperature1->devdata, datasize);
    hipMalloc(&temperature2->devdata, datasize);

    hipMemcpy(temperature1->devdata, temperature1->data, datasize, hipMemcpyHostToDevice);
    hipMemcpy(temperature2->devdata, temperature2->data, datasize, hipMemcpyHostToDevice);
}

/* Copy a temperature field from the device to the host */
void update_host(field *temperature)
{
    size_t datasize;

    datasize = (temperature->nx + 2) * (temperature->ny + 2) * sizeof(double);
    hipMemcpy(temperature->data, temperature->devdata, datasize, hipMemcpyDeviceToHost);
}

/* Copy a temperature field from the host to the device */
void update_device(field *temperature)
{
    size_t datasize;

    datasize = (temperature->nx + 2) * (temperature->ny + 2) * sizeof(double);
    hipMemcpy(temperature->devdata, temperature->data, datasize, hipMemcpyHostToDevice);
}

