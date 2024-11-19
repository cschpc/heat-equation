// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"
#include <Kokkos_Core.hpp>

// Exchange the boundary values
void exchange(Field& field, ParallelData parallel)
{
    double* s_ptr;
    double* r_ptr;
  
    // Send to up, receive from down
    auto sview = Kokkos::subview (field.temperature, 1, Kokkos::ALL);
    auto rview = Kokkos::subview (field.temperature, field.nx + 1, Kokkos::ALL);

    if (parallel.pack_data) {
      if (parallel.nup != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack buffers", field.ny + 2, 
           KOKKOS_LAMBDA(const int i) {
           parallel.sbuf(i) = sview(i);
         });
        Kokkos::fence();
      }
      s_ptr = parallel.sbuf.data();
      r_ptr = parallel.rbuf.data();
    } else {
      s_ptr = sview.data();
      r_ptr = rview.data();
    }

    MPI_Sendrecv(s_ptr, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 11,
                 r_ptr, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (parallel.pack_data && (parallel.ndown != MPI_PROC_NULL)) {
      Kokkos::parallel_for("unpack buffers", field.ny + 2, 
         KOKKOS_LAMBDA(const int i) {
           rview(i) = parallel.rbuf(i);
        });
      Kokkos::fence();
    }

    // Send to down, receive from up
    sview = Kokkos::subview (field.temperature, field.nx, Kokkos::ALL);
    rview = Kokkos::subview (field.temperature, 0, Kokkos::ALL);

    if (parallel.pack_data) {
      if (parallel.ndown != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack buffers", field.ny + 2, 
           KOKKOS_LAMBDA(const int i) {
             parallel.sbuf(i) = field.temperature(field.nx, i);
         });
        Kokkos::fence();
      }
      s_ptr = parallel.sbuf.data();
      r_ptr = parallel.rbuf.data();
    } else {
      s_ptr = sview.data();
      r_ptr = rview.data();
    }

    MPI_Sendrecv(s_ptr, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 12,
                 r_ptr, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (parallel.pack_data && (parallel.nup != MPI_PROC_NULL)) {
      Kokkos::parallel_for("unpack buffers", field.ny + 2, 
         KOKKOS_LAMBDA(const int i) {
           rview(i) = parallel.rbuf(i);
         });
      Kokkos::fence();
    }
}

struct evolveFunctor {
  //Kokkos::View<double**, Kokkos::LayoutLeft> curr; 
  //Kokkos::View<double**, Kokkos::LayoutLeft> prev; 
  Kokkos::View<double**> curr; 
  Kokkos::View<double**> prev; 
  const double a;
  const double dt;
  double inv_dx2, inv_dy2;

  evolveFunctor(Field& curr_, Field& prev_, const double a_, const double dt_) :
     curr(curr_.temperature), prev(prev_.temperature), a(a_), dt(dt_) {
    inv_dx2 = 1.0 / (prev_.dx * prev_.dx);  
    inv_dy2 = 1.0 / (prev_.dy * prev_.dy);  
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
        curr(i, j) = prev(i, j) + a * dt * (
                 ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	         ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
                 );
  };
};

// Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, const double a, const double dt)
{

  using MDPolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
  MDPolicyType mdpolicy({1, 1}, {curr.nx + 1, curr.ny + 1});

  Kokkos::parallel_for("evolve", mdpolicy, evolveFunctor(curr, prev, a, dt));
}
