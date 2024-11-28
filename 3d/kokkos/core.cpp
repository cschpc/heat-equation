// Main solver routines for heat equation solver

#include "parallel.hpp"
#include "heat.hpp"
#include <Kokkos_Core.hpp>

#ifndef NO_MPI
#include <mpi.h>
#endif

// Communicate with manual packing to/from send/receive buffers
void exchange_init(Field& field, ParallelData& parallel)
{
#ifdef NO_MPI
     return;
#endif
    size_t buf_size[3];
    double *sbuf, *rbuf;

    buf_size[0] = (field.ny + 2) * (field.nz + 2);
    buf_size[1] = (field.nx + 2) * (field.nz + 2);
    buf_size[2] = (field.nx + 2) * (field.ny + 2);

    int nreq = 0;
    // Post receives
    for (int d=0; d < 3; d++)
      for (int i=0; i < 2; i++) {
        rbuf = parallel.recv_buffers[d][i].data();
        MPI_Irecv(rbuf, buf_size[d], MPI_DOUBLE,
              parallel.ngbrs[d][i], 11, parallel.comm, &parallel.requests[nreq]);
        nreq++;
      }

    // Copy boundary values to send buffers
    
    // use explicit execution space to enable asynchronous copies
    auto ex = Kokkos::DefaultExecutionSpace();

    // x-direction
    auto sview_1 = Kokkos::subview(field.temperature, 1, Kokkos::ALL, Kokkos::ALL);
    if (parallel.ngbrs[0][0] != MPI_PROC_NULL) 
      Kokkos::deep_copy(ex, parallel.send_buffers[0][0], sview_1);
    auto sview_2 = Kokkos::subview(field.temperature, field.nx, Kokkos::ALL, Kokkos::ALL);
    if (parallel.ngbrs[0][1] != MPI_PROC_NULL) 
      Kokkos::deep_copy(ex, parallel.send_buffers[0][1], sview_2);

    // y-direction
    auto sview_3 = Kokkos::subview(field.temperature, Kokkos::ALL, 1, Kokkos::ALL);
    if (parallel.ngbrs[1][0] != MPI_PROC_NULL) 
      Kokkos::deep_copy(ex, parallel.send_buffers[1][0], sview_3);
    auto sview_4 = Kokkos::subview(field.temperature, Kokkos::ALL, field.ny, Kokkos::ALL);
    if (parallel.ngbrs[1][1] != MPI_PROC_NULL) 
      Kokkos::deep_copy(ex, parallel.send_buffers[1][1], sview_4);

    // z-direction
    auto sview_5 = Kokkos::subview (field.temperature, Kokkos::ALL, Kokkos::ALL, 1);
    if (parallel.ngbrs[2][0] != MPI_PROC_NULL) 
      Kokkos::deep_copy(ex, parallel.send_buffers[1][0], sview_5);
    auto sview_6 = Kokkos::subview (field.temperature, Kokkos::ALL, Kokkos::ALL, field.nz);
    if (parallel.ngbrs[2][1] != MPI_PROC_NULL) 
      Kokkos::deep_copy(ex, parallel.send_buffers[1][1], sview_6);

    // Synchronize copies
    Kokkos::fence();

    // Post sends
    for (int d=0; d < 3; d++)
      for (int i=0; i < 2; i++) {
        sbuf = parallel.send_buffers[d][i].data();
        MPI_Isend(sbuf, buf_size[d], MPI_DOUBLE,
              parallel.ngbrs[d][i], 11, parallel.comm, &parallel.requests[nreq]);
        nreq++;
      }
}

void exchange_finalize(Field& field, ParallelData& parallel)
{
#ifdef NO_MPI
     return;
#endif

    // use explicit execution space to enable asynchronous copies
    auto ex = Kokkos::DefaultExecutionSpace();

    MPI_Waitall(12, parallel.requests, MPI_STATUSES_IGNORE);

    // copy from halos
    // x-direction
    auto rview_1 = Kokkos::subview(field.temperature, 0, Kokkos::ALL, Kokkos::ALL);
    if (parallel.ngbrs[0][0] != MPI_PROC_NULL)
      Kokkos::deep_copy(ex, rview_1, parallel.recv_buffers[0][0]);
        
    auto rview_2 = Kokkos::subview(field.temperature, field.nx + 1, Kokkos::ALL, Kokkos::ALL);
    if (parallel.ngbrs[0][1] != MPI_PROC_NULL)
      Kokkos::deep_copy(ex, rview_2, parallel.recv_buffers[0][1]);

    // y-direction
    auto rview_3 = Kokkos::subview(field.temperature, Kokkos::ALL, 0, Kokkos::ALL);
    if (parallel.ngbrs[1][0] != MPI_PROC_NULL)
      Kokkos::deep_copy(ex, rview_3, parallel.recv_buffers[1][0]);
        
    auto rview_4 = Kokkos::subview(field.temperature, Kokkos::ALL, field.ny + 1, Kokkos::ALL);
    if (parallel.ngbrs[1][1] != MPI_PROC_NULL)
      Kokkos::deep_copy(ex, rview_4, parallel.recv_buffers[1][1]);

    // z-direction
    auto rview_5 = Kokkos::subview(field.temperature, Kokkos::ALL, Kokkos::ALL, 0);
    if (parallel.ngbrs[2][0] != MPI_PROC_NULL)
      Kokkos::deep_copy(ex, rview_5, parallel.recv_buffers[2][0]);
        
    auto rview_6 = Kokkos::subview(field.temperature, Kokkos::ALL, Kokkos::ALL, field.nz + 1);
    if (parallel.ngbrs[2][1] != MPI_PROC_NULL)
      Kokkos::deep_copy(ex, rview_6, parallel.recv_buffers[2][1]);

    // Synchronize copies
    Kokkos::fence();
}


// Exchange the boundary values
void exchange(Field& field, ParallelData& parallel)
{
#ifdef NO_MPI
    return;
#else
    exchange_init(field, parallel);
    exchange_finalize(field, parallel);
#endif
}
    

// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);

  // Direct "View"s are needed for the lambda
  auto curr_temp = curr.temperature;
  auto prev_temp = prev.temperature;

  using MDPolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<3> >;
  MDPolicyType mdpolicy({1, 1, 1}, {curr.nx + 1, curr.ny + 1, curr.nz + 1});

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  Kokkos::parallel_for("evolve", mdpolicy,
     KOKKOS_LAMBDA(const int i, const int j, const int k) {
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
      });

  Kokkos::fence();

}

void evolve_interior(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);

  // Direct "View"s are needed for the lambda
  auto curr_temp = curr.temperature;
  auto prev_temp = prev.temperature;

  using MDPolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<3> >;
  MDPolicyType mdpolicy({2, 2, 2}, {curr.nx, curr.ny, curr.nz});

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  Kokkos::parallel_for("evolve_interior", mdpolicy,
     KOKKOS_LAMBDA(const int i, const int j, const int k) {
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
      });

}

void evolve_edges(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);
  auto inv_dz2 = 1.0 / (prev.dz * prev.dz);

  // Direct "View"s are needed for the lambda
  auto curr_temp = curr.temperature;
  auto prev_temp = prev.temperature;

  using MDPolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;

  MDPolicyType mdpolicy_z({1, 1}, {curr.nx + 1, curr.ny + 1});

  Kokkos::parallel_for("evolve_z_edges", mdpolicy_z,
     KOKKOS_LAMBDA(const int i, const int j) {
            int k = 1;
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
            k = curr.nz;
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
      });

  MDPolicyType mdpolicy_y({1, 1}, {curr.nx + 1, curr.nz + 1});

  Kokkos::parallel_for("evolve_y_edges", mdpolicy_y,
     KOKKOS_LAMBDA(const int i, const int k) {
            int j = 1;
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
            j = curr.ny;
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
      });

  MDPolicyType mdpolicy_x({1, 1}, {curr.ny + 1, curr.nz + 1});

  Kokkos::parallel_for("evolve_x_edges", mdpolicy_x,
     KOKKOS_LAMBDA(const int j, const int k) {
            int i = 1;
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
            i = curr.nx;
            curr_temp(i, j, k) = prev_temp(i, j, k) + a * dt * (
	        ( prev_temp(i + 1, j, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i - 1, j, k) ) * inv_dx2 +
	        ( prev_temp(i, j + 1, k) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j - 1, k) ) * inv_dy2 +
	        ( prev_temp(i, j, k + 1) - 2.0 * prev_temp(i, j, k) + prev_temp(i, j, k - 1) ) * inv_dz2 
               );
      });

  Kokkos::fence();

}
