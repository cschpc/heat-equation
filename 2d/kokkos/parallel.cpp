// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include "heat.hpp"
#include <mpi.h>
#include <Kokkos_Core.hpp>

ParallelData::ParallelData() {

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nup = rank - 1;
    ndown = rank + 1;

    if (nup < 0) {
        nup = MPI_PROC_NULL;
    }
    if (ndown > size - 1) {
        ndown = MPI_PROC_NULL;
    }

    // Check if default layout is Left
    pack_data = std::is_same_v<Kokkos::DefaultExecutionSpace::array_layout, 
                               Kokkos::LayoutLeft>;
}

void ParallelData::set_buffers(const int ny) {

  // Allocate buffers for LayoutLeft
  if (pack_data) { 
    sbuf = Kokkos::View<double*>("send buffer", ny + 2); 
    rbuf = Kokkos::View<double*>("recv buffer", ny + 2); 
  }
}
