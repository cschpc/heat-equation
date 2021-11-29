// Main solver routines for heat equation solver

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "heat.hpp"

// Update the temperature values using five-point stencil 
__global__ void evolve_kernel(double *currdata, const double *prevdata, double a, double dt, int nx,                              int ny, int nz, double inv_dx2, double inv_dy2, double inv_dz2)
{


    // CUDA / HIP threads are arranged in column major order; thus k index from x, j from y ...
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    if (i > 0 && j > 0 && i < nx+1 && j < ny+1 && k > 0 && k < nz+1) {
      int ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
      int ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
      int im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
      int jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
      int jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
      int kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
      int km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);
      currdata[ind] = prevdata[ind] + a*dt*
           ((prevdata[ip] - 2.0*prevdata[ind] + prevdata[im]) * inv_dx2 +
            (prevdata[jp] - 2.0*prevdata[ind] + prevdata[jm]) * inv_dy2 +
            (prevdata[kp] - 2.0*prevdata[ind] + prevdata[km]) * inv_dz2);
    }

}


// Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);

  int nx = curr.nx;
  int ny = curr.ny;
  int nz = curr.nz;

  double *currdata = curr.devdata;
  const double *prevdata = prev.devdata;
  size_t field_size = (curr.nx + 2) * (curr.ny + 2) * (curr.nz + 2);

  // HIP thread settings */
  constexpr int blocksize = 8;  //!< HIP thread block dimension
  dim3 dimBlock(blocksize, blocksize, blocksize);
  // HIP threads are arranged in column major order; thus make nz x ny x nx grid
  dim3 dimGrid((nz + 2 + blocksize - 1) / blocksize,
	       (ny + 2 + blocksize - 1) / blocksize,
               (nx + 2 + blocksize - 1) / blocksize);

  hipLaunchKernelGGL(evolve_kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, currdata, prevdata, 
		     a, dt, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2);
  hipDeviceSynchronize();
  

}

// Start a data region and copy temperature fields to the device
void enter_data(Field& curr, Field& prev)
{

    double *currdata = curr.temperature.data();
    double *prevdata = prev.temperature.data();
    size_t field_size = (curr.nx + 2) * (curr.ny + 2) * (curr.nz + 2);

    hipMalloc(&curr.devdata, field_size);    
    hipMalloc(&prev.devdata, field_size);    

    hipMemcpy(curr.devdata, currdata, field_size, hipMemcpyHostToDevice);
    hipMemcpy(prev.devdata, prevdata, field_size, hipMemcpyHostToDevice);
}

// End a data region and copy temperature fields back to the host
void exit_data(Field& curr, Field& prev)
{
    double *currdata = curr.temperature.data();
    double *prevdata = prev.temperature.data();
    size_t field_size = (curr.nx + 2) * (curr.ny + 2) * (curr.nz + 2);

    hipMemcpy(currdata, curr.devdata, field_size, hipMemcpyDeviceToHost);
    hipMemcpy(prevdata, prev.devdata, field_size, hipMemcpyDeviceToHost);

    hipFree(curr.devdata);
    hipFree(prev.devdata);
}

// Copy a temperature field from the device to the host 
void update_host(Field& temperature)
{

    double *data = temperature.temperature.data();
    double *devdata = temperature.devdata;
    size_t field_size = (temperature.nx + 2) * (temperature.ny + 2) * (temperature.nz + 2);

    hipMemcpy(data, devdata, field_size, hipMemcpyDeviceToHost);
}

// Copy a temperature field from the host to the device
void update_device(Field& temperature)
{

    double *data = temperature.temperature.data();
    double *devdata = temperature.devdata;
    size_t field_size = (temperature.nx + 2) * (temperature.ny + 2) * (temperature.nz + 2);

    hipMemcpy(devdata, data, field_size, hipMemcpyHostToDevice);
}

