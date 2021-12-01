// Main solver routines for heat equation solver

#ifndef NO_MPI
#include <mpi.h>
#endif

#include "heat.hpp"
#include "parallel.hpp"

// Exchange the boundary values
void exchange(Field& field, ParallelData& parallel)
{
#ifdef NO_MPI
    return;
#else    

    size_t buf_size;
    double *sbuf, *rbuf;
#ifdef MPI_DATATYPES
    // x-direction
    sbuf = field.temperature.data(1, 0, 0);
    rbuf = field.temperature.data(field.nx + 1, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][0], 11, parallel.comm, &parallel.requests[0]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][1], 11, parallel.comm, &parallel.requests[1]);
    
    sbuf = field.temperature.data(field.nx, 0, 0);
    rbuf = field.temperature.data(0, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][1], 12, parallel.comm, &parallel.requests[2]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][0], 12, parallel.comm, &parallel.requests[3]);
    
    // y-direction
    sbuf = field.temperature.data(0, 1, 0);
    rbuf = field.temperature.data(0, field.ny + 1, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][0], 21, parallel.comm, &parallel.requests[4]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][1], 21, parallel.comm, &parallel.requests[5]);
    
    sbuf = field.temperature.data(0, field.ny, 0);
    rbuf = field.temperature.data(0, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][1], 22, parallel.comm, &parallel.requests[6]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][0], 22, parallel.comm, &parallel.requests[7]);
  
    // z-direction
    sbuf = field.temperature.data(0, 0, 1);
    rbuf = field.temperature.data(0, 0, field.nz + 1);
    MPI_Isend(sbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][0], 31, parallel.comm, &parallel.requests[8]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][1], 31, parallel.comm, &parallel.requests[9]);
    
    sbuf = field.temperature.data(0, 0, field.nz);
    rbuf = field.temperature.data(0, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][1], 32, parallel.comm, &parallel.requests[10]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][0], 32, parallel.comm, &parallel.requests[11]);
    
    MPI_Waitall(12, parallel.requests, MPI_STATUSES_IGNORE);
#elif defined MPI_NEIGHBORHOOD
    MPI_Datatype types[6] = {parallel.halotypes[0], parallel.halotypes[0],
                             parallel.halotypes[1], parallel.halotypes[1],
                             parallel.halotypes[2], parallel.halotypes[2]};
    int counts[6] = {1, 1, 1, 1, 1, 1};
    MPI_Aint sdisps[6], rdisps[6], disp0;

    // Determine displacements
    disp0 = reinterpret_cast<MPI_Aint> (field.temperature.data());
    sdisps[0] =  reinterpret_cast<MPI_Aint> (field.temperature.data(1, 0, 0));        
    sdisps[1] =  reinterpret_cast<MPI_Aint> (field.temperature.data(field.nx, 0, 0)); 
    sdisps[2] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 1, 0));        
    sdisps[3] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, field.ny, 0)); 
    sdisps[4] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 0, 1));        
    sdisps[5] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 0, field.nz)); 

    rdisps[0] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 0, 0));            
    rdisps[1] =  reinterpret_cast<MPI_Aint> (field.temperature.data(field.nx + 1, 0, 0)); 
    rdisps[2] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 0, 0));            
    rdisps[3] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, field.ny + 1, 0)); 
    rdisps[4] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 0, 0));            
    rdisps[5] =  reinterpret_cast<MPI_Aint> (field.temperature.data(0, 0, field.nz + 1)); 

    for (int i=0; i < 6; i++) {
      sdisps[i] -= disp0;
      rdisps[i] -= disp0;
    }

    MPI_Neighbor_alltoallw(field.temperature.data(), counts, sdisps, types,
                           field.temperature.data(), counts, rdisps, types,
                           parallel.comm);

#else
    // x-direction
    buf_size = (field.ny + 2) * (field.nz + 2);
    // copy to halo
    for (int j=0; j < field.ny + 2; j++)
      for (int k=0; k < field.nz + 2; k++) {
         parallel.send_buffers[0][0](j, k) = field.temperature(1, j, k);
         parallel.send_buffers[0][1](j, k) = field.temperature(field.nx, j, k);
      }

    sbuf = parallel.send_buffers[0][0].data();
    rbuf = parallel.recv_buffers[0][0].data();
    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][0], 11, parallel.comm, &parallel.requests[0]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][0], 11, parallel.comm, &parallel.requests[1]);

    sbuf = parallel.send_buffers[0][1].data();
    rbuf = parallel.recv_buffers[0][1].data();
    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][1], 11, parallel.comm, &parallel.requests[2]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][1], 11, parallel.comm, &parallel.requests[3]);


    // y-direction
    buf_size = (field.nx + 2) * (field.nz + 2);
    // copy to halo
    for (int i=0; i < field.nx + 2; i++)
      for (int k=0; k < field.nz + 2; k++) {
         parallel.send_buffers[1][0](i, k) = field.temperature(i, 1, k);
         parallel.send_buffers[1][1](i, k) = field.temperature(i, field.ny, k);
      }

    sbuf = parallel.send_buffers[1][0].data();
    rbuf = parallel.recv_buffers[1][0].data();
    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][0], 11, parallel.comm, &parallel.requests[4]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][0], 11, parallel.comm, &parallel.requests[5]);

    sbuf = parallel.send_buffers[1][1].data();
    rbuf = parallel.recv_buffers[1][1].data();
    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][1], 11, parallel.comm, &parallel.requests[6]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][1], 11, parallel.comm, &parallel.requests[7]);

    // z-direction
    buf_size = (field.nx + 2) * (field.ny + 2);
    // copy to halo
    for (int i=0; i < field.nx + 2; i++)
      for (int j=0; j < field.ny + 2; j++) {
         parallel.send_buffers[2][0](i, j) = field.temperature(i, j, 1);
         parallel.send_buffers[2][1](i, j) = field.temperature(i, j, field.nz);
      }

    sbuf = parallel.send_buffers[2][0].data();
    rbuf = parallel.recv_buffers[2][0].data();
    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][0], 11, parallel.comm, &parallel.requests[8]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][0], 11, parallel.comm, &parallel.requests[9]);

    sbuf = parallel.send_buffers[2][1].data();
    rbuf = parallel.recv_buffers[2][1].data();
    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][1], 11, parallel.comm, &parallel.requests[10]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][1], 11, parallel.comm, &parallel.requests[11]);

    MPI_Waitall(12, parallel.requests, MPI_STATUSES_IGNORE);

    // copy from halos
    // x-direction
    if (parallel.ngbrs[0][0] != MPI_PROC_NULL)
      for (int j=0; j < field.ny + 2; j++)
        for (int k=0; k < field.nz + 2; k++) {
          field.temperature(0, j, k) = parallel.recv_buffers[0][0](j, k);
        }  
    if (parallel.ngbrs[0][1] != MPI_PROC_NULL)
      for (int j=0; j < field.ny + 2; j++)
        for (int k=0; k < field.nz + 2; k++) {
          field.temperature(field.nx + 1, j, k) = parallel.recv_buffers[0][1](j, k);
        }

    // y-direction
    if (parallel.ngbrs[1][0] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
        for (int k=0; k < field.nz + 2; k++) {
          field.temperature(i, 0, k) = parallel.recv_buffers[1][0](i, k);
      }
    if (parallel.ngbrs[1][1] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
        for (int k=0; k < field.nz + 2; k++) {
          field.temperature(i, field.ny + 1, k) = parallel.recv_buffers[1][1](i, k);
        }

    // z-direction
    if (parallel.ngbrs[2][0] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
        for (int j=0; j < field.ny + 2; j++) {
          field.temperature(i, j, 0) = parallel.recv_buffers[2][0](i, j);
        }
    if (parallel.ngbrs[2][1] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
        for (int j=0; j < field.ny + 2; j++) {
          field.temperature(i, j, field.nz + 1) = parallel.recv_buffers[2][1](i, j);
        }

#endif // MPI_DATATYPES

#endif    
}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);

  auto dx2 = (prev.dx * prev.dx);
  auto dy2 = (prev.dy * prev.dy);
  auto dz2 = (prev.dz * prev.dz);

  int nx = curr.nx;
  int ny = curr.ny;
  int nz = curr.nz;

  double *currdata = curr.temperature.data();
  const double *prevdata = prev.temperature.data();
  size_t field_size = (curr.nx + 2) * (curr.ny + 2) * (curr.nz + 2);
  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
#ifdef DOMP_LOOP
  #pragma omp target loop collapse(3) \
   map(tofrom:currdata[0:field_size], prevdata[0:field_size])
#else
  #pragma omp target teams distribute parallel for simd collapse(3) \
   map(tofrom:currdata[0:field_size], prevdata[0:field_size])
#endif
  for (int i = 1; i < nx + 1; i++) {
    for (int j = 1; j < ny + 1; j++) {
      for (int k = 1; k < nz + 1; k++) {
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
  }

}

// Start a data region and copy temperature fields to the device
void enter_data(Field& curr, Field& prev)
{

    double *currdata = curr.temperature.data();
    double *prevdata = prev.temperature.data();
    size_t field_size = (curr.nx + 2) * (curr.ny + 2) * (curr.nz + 2);

#pragma omp target enter data \
    map(to: currdata[0:field_size], prevdata[0:field_size])
}

// End a data region and copy temperature fields back to the host
void exit_data(Field& curr, Field& prev)
{
    double *currdata = curr.temperature.data();
    double *prevdata = prev.temperature.data();
    size_t field_size = (curr.nx + 2) * (curr.ny + 2) * (curr.nz + 2);

#pragma omp target exit data \
    map(from: currdata[0:field_size], prevdata[0:field_size])
}

// Copy a temperature field from the device to the host 
void update_host(Field& temperature)
{

    double *data = temperature.temperature.data();
    size_t field_size = (temperature.nx + 2) * (temperature.ny + 2) * (temperature.nz + 2);

#pragma omp target update from(data[0:field_size])
}

// Copy a temperature field from the host to the device
void update_device(Field& temperature)
{

    double *data = temperature.temperature.data();
    size_t field_size = (temperature.nx + 2) * (temperature.ny + 2) * (temperature.nz + 2);

#pragma omp target update to(data[0:field_size])
}

