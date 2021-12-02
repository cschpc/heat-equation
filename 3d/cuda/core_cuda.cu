// Main solver routines for heat equation solver
#include "heat.hpp"
#include <cuda_runtime.h>
#include "error_checks.h"
#include <iostream>

// Update the temperature values using five-point stencil */
__global__ void evolve_kernel(double *currdata, double *prevdata, double a, double dt, int nx, int ny, int nz,
                       double inv_dx2, double inv_dy2, double inv_dz2)
{

    // CUDA threads are arranged in column major order; thus k index from x, j from y, ...
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && j > 0 && k > 0 && i < nx+1 && j < ny+1 && k < nz+1) {
      int ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
      int ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
      int im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
      int jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
      int jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
      int kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
      int km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);
      currdata[ind] = prevdata[ind] + a * dt * (
                  ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
                  ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
                  ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
      );
    }
}

void evolve(Field& curr, Field& prev, const double a, const double dt)
{

  int nx = curr.nx;
  int ny = curr.ny;
  int nz = curr.nz;

  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);
  
  // CUDA thread settings 
  const int blocksizes[3] = {64, 4, 4};  //!< CUDA thread block dimension
  dim3 dimBlock(blocksizes[0], blocksizes[1], blocksizes[2]);
   // CUDA threads are arranged in column major order; thus make ny x nx grid
  dim3 dimGrid((nz + 2 + blocksizes[0] - 1) / blocksizes[0],
               (ny + 2 + blocksizes[1] - 1) / blocksizes[1],
               (nx + 2 + blocksizes[2] - 1) / blocksizes[2]);

#ifdef CUDA_MANAGED
  auto currdata = curr.temperature.data();
  auto prevdata = prev.temperature.data();
#else
  auto currdata = curr.temperature_dev;
  auto prevdata = prev.temperature_dev;
#endif

  evolve_kernel<<<dimGrid, dimBlock>>>(currdata, prevdata, a, dt, nx, ny, nz, 
                                         inv_dx2, inv_dy2, inv_dz2);
  cudaDeviceSynchronize();
  CHECK_ERROR_MSG("evolve");

}

void allocate_data(Field& field1, Field& field2)
{
#ifdef CUDA_MANAGED
    return;
#else
    size_t field_size = (field1.nx + 2) * (field1.ny + 2) * (field1.nz + 2) * sizeof(double);
    GPU_CHECK( cudaMalloc(&field1.temperature_dev, field_size) ); 
    GPU_CHECK( cudaMalloc(&field2.temperature_dev, field_size) );
#endif
}

void enter_data(Field& field1, Field& field2)
{
    size_t field_size = (field1.nx + 2) * (field1.ny + 2) * (field1.nz + 2) * sizeof(double);

#ifdef CUDA_MANAGED
    cudaMemPrefetchAsync(field1.devdata(), field_size, 0);
    cudaMemPrefetchAsync(field2.devdata(), field_size, 0);
#else
    GPU_CHECK( cudaMemcpy(field1.temperature_dev, field1.temperature.data(), field_size, cudaMemcpyHostToDevice) );
    GPU_CHECK( cudaMemcpy(field2.temperature_dev, field2.temperature.data(), field_size, cudaMemcpyHostToDevice) );
#endif
}

void exit_data(Field& field1, Field& field2)
{
#ifdef CUDA_MANAGED
    return;
#else
    size_t field_size = (field1.nx + 2) * (field1.ny + 2) * (field1.nz + 2) * sizeof(double);

    cudaMemcpy(field1.temperature.data(), field1.temperature_dev, field_size, cudaMemcpyDeviceToHost) ;
    GPU_CHECK( cudaMemcpy(field2.temperature.data(), field2.temperature_dev, field_size, cudaMemcpyDeviceToHost) );
#endif
}

void free_data(Field& field1, Field& field2)
{

#ifdef CUDA_MANAGED
    return;
#else
    GPU_CHECK( cudaFree(field1.temperature_dev) );
    GPU_CHECK( cudaFree(field2.temperature_dev) );
#endif
}

/* Copy a temperature field from the device to the host */
void update_host(Field& field)
{
#ifdef CUDA_MANAGED
    return;
#else
    size_t field_size = (field.nx + 2) * (field.ny + 2) * (field.nz + 2) * sizeof(double);

    GPU_CHECK( cudaMemcpy(field.temperature.data(), field.temperature_dev, field_size, cudaMemcpyDeviceToHost) );
#endif
}
/* Copy a temperature field from the host to the device */
void update_device(Field& field)
{
#ifdef CUDA_MANAGED
    return;
#else
    size_t field_size = (field.nx + 2) * (field.ny + 2) * (field.nz + 2) * sizeof(double);
    GPU_CHECK( cudaMemcpy(field.temperature_dev, field.temperature.data(), field_size, cudaMemcpyHostToDevice) );
#endif
}

